//! An FTDC archive: concatenated BSON documents, read in file order.
//!
//! An archive (`metrics.<utc>-<uniquifier>`, or `metrics.interim`) is not
//! one BSON document. It is a sequence of them, and `bsondump` shows each
//! one. Every document begins with the usual BSON length, a little-endian
//! `int32` that counts itself. This reader pulls that length, then exactly
//! the remaining bytes, then parses the document. A file that ends on a
//! document boundary is complete. A file that ends inside a length or
//! inside a document is truncated, and that is an error: the previous
//! reader treated any parse failure at EOF as a clean end, so a chopped
//! last document disappeared.
//!
//! The `type` field says which kind of payload follows:
//!
//! * `0` — metadata, written when the file is opened. The payload is `doc`
//!   and describes the host and the collectors. There is normally one.
//! * `1` — a metric chunk. The payload is `data`, a BinData blob. This is
//!   the only type that carries the time series. See `decode`.
//! * `2` — a periodic metadata delta. MongoDB 7 and later write these for
//!   slow-changing values such as server parameters. The payload is `doc`
//!   plus a `count`. A metrics file usually has one of these before the
//!   first metric chunk. They are not samples.
//!
//! The server has grown this set before (`2` itself was the addition). A
//! type this build does not know is skipped and remembered, so metric
//! chunks that follow it are still decoded.
//!
//! Inflation and delta reconstruction have no cross-chunk dependency, so
//! each chunk runs on its own thread. The output does. Sample 0 of chunk
//! N+1 is the sample after the last sample of chunk N. The handles are
//! joined in the order they were spawned, and only then appended. A later
//! chunk that finishes first waits its turn. Pushing into a shared `Vec`
//! from whichever thread took the lock first scrambled that order, and it
//! scrambled it differently on every run. A worker's `Err` used to be
//! discarded with `let _ =`, so a chunk that failed to inflate, or whose
//! `metric_count` did not match the reference walk, disappeared and the
//! file still looked successful.
//!
//! Rust requires the `impl` of [`FTDCFileDecoder`] to live in this module.
//! The methods here are the public entry points. The bytes they interpret
//! are decoded by `varint`, `extract`, `decode`, and `export`.

use crate::decode::{append_chunk, decode_one_chunk, inflate_owned};
use crate::errors::FileDecoderError;
use crate::export::write_metrics_parquet;
use bson::{Binary, Bson, Document};
use indexmap::IndexMap;
use regex::Regex;
use std::io::{BufReader, Read};
use std::path::Path;
use std::thread;

/// `type` values on the BSON documents concatenated inside an FTDC archive.
const TYPE_METADATA: i32 = 0;
const TYPE_METRIC_CHUNK: i32 = 1;
const TYPE_METADATA_DELTA: i32 = 2;

/// A struct that decodes an FTDC file.
///
/// The compressed metric blobs live here until [`FTDCFileDecoder::uncompress_data`]
/// inflates them. Metadata documents are kept whole so a later export can
/// write them as JSON without parsing `Document`'s `Display` text back apart.
pub struct FTDCFileDecoder {
    /// Type 0 `doc` sub-documents, in file order.
    pub(crate) metadata_docs: Vec<Document>,
    /// Type 2 `doc` sub-documents, in file order.
    ///
    /// The old field name for this was `parameters`. Type 2 is a metadata
    /// delta, not the startup parameter block. Several of them can appear
    /// in one file; keeping only the last one drops the earlier samples.
    pub(crate) metadata_deltas: Vec<Document>,
    /// BSON documents visited while reading, of every type, including
    /// types that were skipped.
    pub(crate) document_count: usize,
    /// `type` numbers this build did not recognise. Reading continued.
    pub(crate) skipped_chunk_types: Vec<i32>,
    /// Type 1 `data` blobs, still compressed, in file order.
    pub(crate) binary_chunks: Vec<Binary>,
}

/// Read the next BSON document from a concatenation.
///
/// `Ok(None)` is a clean end: the read of the length returned zero bytes.
/// A short length, a length below 5 (the int32 itself plus the trailing
/// NUL; this also rejects the negative numbers a corrupt file can spell),
/// or a document body that ends early is reported. The length is what
/// sizes the buffer, which is how a BSON reader consumes a framing stream.
fn read_length_prefixed_document<R: Read>(
    reader: &mut R,
) -> Result<Option<Document>, FileDecoderError> {
    let mut len_buf = [0u8; 4];
    let mut filled = 0usize;
    while filled < 4 {
        match reader.read(&mut len_buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => {
                return Err(FileDecoderError::format(format!(
                    "FTDC archive ends in the middle of a BSON document length ({filled} of 4 bytes)"
                )));
            }
            Ok(n) => filled += n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(FileDecoderError::format(format!(
                    "failed to read FTDC archive: {error}"
                )));
            }
        }
    }

    let doc_len = i32::from_le_bytes(len_buf);
    if doc_len < 5 {
        return Err(FileDecoderError::format(format!(
            "BSON document length {doc_len} is smaller than 5"
        )));
    }
    let doc_len = doc_len as usize;

    let mut body = vec![0u8; doc_len];
    body[..4].copy_from_slice(&len_buf);
    if let Err(error) = reader.read_exact(&mut body[4..]) {
        return Err(match error.kind() {
            std::io::ErrorKind::UnexpectedEof => FileDecoderError::format(format!(
                "FTDC archive ends in the middle of a BSON document of {doc_len} bytes"
            )),
            _ => FileDecoderError::format(format!("failed to read FTDC archive: {error}")),
        });
    }

    let mut cursor = body.as_slice();
    Document::from_reader(&mut cursor)
        .map(Some)
        .map_err(|error| {
            FileDecoderError::format(format!(
                "FTDC archive contains a BSON document that does not parse: {error}"
            ))
        })
}

impl FTDCFileDecoder {
    pub const INITIAL_METRIC_CAPACITY: usize = 10240;

    /// Creates a new structure to decode a single file.
    pub fn new() -> FTDCFileDecoder {
        FTDCFileDecoder {
            metadata_docs: Vec::new(),
            metadata_deltas: Vec::new(),
            document_count: 0,
            skipped_chunk_types: Vec::new(),
            binary_chunks: vec![],
        }
    }

    /// Reads an FTDC metrics file, storing the BSON documents in the structure.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The file to read
    pub fn read_file(&mut self, file_name: impl AsRef<Path>) -> Result<(), FileDecoderError> {
        let file_name = file_name.as_ref();
        let file = match std::fs::File::open(file_name) {
            Ok(file) => file,
            Err(error) => return Err(FileDecoderError::io("failed to open", file_name, error)),
        };
        // The archive is read front to back, one document at a time. The
        // buffer only cuts down the number of reads; each document's own
        // length is what decides how many bytes belong to it.
        let mut reader = BufReader::new(file);

        loop {
            match read_length_prefixed_document(&mut reader)? {
                None => break,
                Some(metrics) => {
                    // Count this document even when we skip it, so
                    // `document_count` describes the file rather than
                    // only the documents we knew how to use.
                    self.document_count += 1;
                    match chunk_type_of(&metrics)? {
                        TYPE_METADATA => {
                            self.metadata_docs
                                .push(embedded_doc(&metrics, "doc", "metadata")?);
                        }
                        TYPE_METRIC_CHUNK => {
                            self.binary_chunks.push(metric_blob(&metrics)?);
                        }
                        TYPE_METADATA_DELTA => {
                            // Keep every delta. A file from MongoDB 7+
                            // has one of these before the first metric
                            // chunk, and a long-running process appends
                            // more as parameters change. Replacing the
                            // previous value would leave only the last.
                            self.metadata_deltas.push(embedded_doc(
                                &metrics,
                                "doc",
                                "metadata delta",
                            )?);
                        }
                        other => {
                            // Do not abort. The metric chunks after an
                            // unknown type are still valid samples, and
                            // failing here is how a whole archive used
                            // to be discarded because of one document.
                            log::warn!(
                                "skipping unrecognised FTDC chunk type {other}; metric chunks around it are still read"
                            );
                            self.skipped_chunk_types.push(other);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Inflate every metric chunk, and return the chunks in file order.
    ///
    /// See the module docs for why the join order is the file order.
    pub fn uncompress_data(&self) -> Result<Vec<Vec<u8>>, FileDecoderError> {
        let mut handles = Vec::with_capacity(self.binary_chunks.len());
        for blob in &self.binary_chunks {
            let bytes = blob.bytes.clone();
            handles.push(thread::spawn(move || inflate_owned(bytes)));
        }

        let mut inflated = Vec::with_capacity(handles.len());
        for handle in handles {
            let chunk = handle
                .join()
                .map_err(|_| FileDecoderError::format(format!("FTDC inflate thread panicked")))??;
            inflated.push(chunk);
        }
        Ok(inflated)
    }

    /// Reads an FTDC file and decompresses the data bearing documents in it.
    pub fn read_and_uncompress(
        &mut self,
        file_name: impl AsRef<Path>,
    ) -> Result<Vec<Vec<u8>>, FileDecoderError> {
        self.read_file(file_name)?;
        self.uncompress_data()
    }

    /// Decode inflated chunks and append each metric in file order.
    ///
    /// Each chunk is decoded on its own thread, because the reference
    /// walk and the delta reconstruction do not look at other chunks.
    /// The append does. A metric's samples are the concatenation of its
    /// samples from chunk 0, then chunk 1, then chunk 2. Joining the
    /// workers in spawn order and appending only after the previous
    /// chunk has been appended is what keeps that concatenation in time
    /// order.
    ///
    /// A worker's `Err` stops the decode.
    pub fn decode_decompressed_data_chunks(
        &self,
        data_chunks: Vec<Vec<u8>>,
    ) -> Result<IndexMap<String, Vec<Option<u64>>>, FileDecoderError> {
        let mut handles = Vec::with_capacity(data_chunks.len());
        for chunk in data_chunks {
            handles.push(thread::spawn(move || decode_one_chunk(chunk)));
        }

        let mut merged: IndexMap<String, Vec<Option<u64>>> = IndexMap::new();
        // How many samples the chunks already appended account for.
        // A metric that first appears later is null for these rows.
        let mut rows = 0usize;
        for handle in handles {
            let metrics = handle
                .join()
                .map_err(|_| FileDecoderError::format(format!("FTDC decode thread panicked")))??;
            append_chunk(&mut merged, &mut rows, metrics)?;
        }
        Ok(merged)
    }

    /// Reads an FTDC file, uncompresses binary docs contained in it and decodes deltas.
    ///
    /// # Arguments
    ///
    /// * `file_name` -  The name of the file to decode.
    ///
    /// # Returns
    ///
    /// * Metric names in the order they were first seen, each with its samples.
    ///   That is the reference document's field order, then metrics introduced
    ///   by later chunks.
    pub fn read_and_decode(
        &mut self,
        file_name: impl AsRef<Path>,
        filter: Option<Regex>,
    ) -> Result<IndexMap<String, Vec<Option<u64>>>, FileDecoderError> {
        let data_chunks = self.read_and_uncompress(file_name)?;
        let decoded_data = self.decode_decompressed_data_chunks(data_chunks)?;

        match filter {
            Some(re) => {
                let mut matched_data: IndexMap<String, Vec<Option<u64>>> = IndexMap::new();

                for (key, value) in decoded_data {
                    if re.is_match(&key) {
                        matched_data.insert(key, value);
                    }
                }
                Ok(matched_data)
            }
            None => Ok(decoded_data),
        }
    }

    /// Build a DataFrame whose columns are the decoded series.
    pub fn metrics_to_dataframe(
        metrics: IndexMap<String, Vec<Option<u64>>>,
    ) -> Result<polars::prelude::DataFrame, FileDecoderError> {
        crate::export::metrics_to_dataframe(metrics)
    }

    /// Decodes an FTDC file and writes a Parquet file.
    ///
    /// `metadata`, when set, is the path of the sidecar. It receives every
    /// type 0 document and every type 2 document as relaxed extended JSON.
    pub fn write_to_parquet(
        input_file_name: impl AsRef<Path>,
        output_file_name: impl AsRef<Path>,
        metadata: Option<&Path>,
        filter: Option<Regex>,
    ) -> Result<(), FileDecoderError> {
        let mut parsed_file = FTDCFileDecoder::new();
        let metrics = parsed_file.read_and_decode(input_file_name, filter)?;
        write_metrics_parquet(
            metrics,
            output_file_name.as_ref(),
            metadata,
            &parsed_file.metadata_docs,
            &parsed_file.metadata_deltas,
            parsed_file.document_count,
            &parsed_file.skipped_chunk_types,
        )
    }
}

/// Read the chunk `type` without panicking when it is missing or not an int.
///
/// The server writes it with `appendNumber`, which is an int32 for these
/// small values. int64 is accepted too, because a permissive writer (and
/// some round-trips through extended JSON) will widen it.
fn chunk_type_of(doc: &Document) -> Result<i32, FileDecoderError> {
    match doc.get("type") {
        Some(Bson::Int32(value)) => Ok(*value),
        Some(Bson::Int64(value)) => i32::try_from(*value).map_err(|_| {
            FileDecoderError::format(format!("FTDC chunk type {value} does not fit in an int32"))
        }),
        Some(other) => Err(FileDecoderError::format(format!(
            "FTDC chunk type is {other:?}, expected an int"
        ))),
        None => Err(FileDecoderError::format(format!(
            "'type' key not found in chunk"
        ))),
    }
}

/// Take the embedded document stored under `key`.
///
/// Type 0 and type 2 both keep their payload in `doc`. A chunk that claims
/// to be one of those and has no document is corrupt; unlike an unknown
/// type number, there is nothing sensible to skip to.
fn embedded_doc(doc: &Document, key: &str, kind: &str) -> Result<Document, FileDecoderError> {
    match doc.get(key) {
        Some(Bson::Document(inner)) => Ok(inner.clone()),
        Some(other) => Err(FileDecoderError::format(format!(
            "'{key}' in an FTDC {kind} chunk is {other:?}, expected a document"
        ))),
        None => Err(FileDecoderError::format(format!(
            "'{key}' key not found in an FTDC {kind} chunk"
        ))),
    }
}

/// Take the compressed blob from a type 1 chunk.
///
/// A metric chunk whose `data` is not BinData used to be ignored, which
/// deleted a block of samples and shifted every later timestamp. That is a
/// broken file, so it is reported.
fn metric_blob(doc: &Document) -> Result<Binary, FileDecoderError> {
    match doc.get("data") {
        Some(Bson::Binary(blob)) => Ok(blob.clone()),
        Some(other) => Err(FileDecoderError::format(format!(
            "'data' in an FTDC metric chunk is {other:?}, expected BinData"
        ))),
        None => Err(FileDecoderError::format(format!(
            "'data' key not found in an FTDC metric chunk"
        ))),
    }
}
