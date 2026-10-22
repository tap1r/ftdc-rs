extern crate core;
mod errors;

use crate::errors::FileDecoderError;
use anyhow::anyhow;
use bson::{Binary, Bson, Document};
use buffered_reader;
use buffered_reader::BufferedReader;
use polars::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::thread;
use zune_inflate::DeflateDecoder;

/// `type` values on the BSON documents concatenated inside an FTDC archive.
///
/// An archive (`metrics.<utc>-<uniquifier>`, or `metrics.interim`) is not one
/// BSON document. It is a sequence of them. `bsondump` shows each one. The
/// `type` field says which kind of payload follows:
///
/// * `0` — metadata, written when the file is opened. The payload is `doc`
///   and describes the host and the collectors. There is normally one.
/// * `1` — a metric chunk. The payload is `data`, a BinData blob: a `uint32`
///   uncompressed length, then a zlib stream of deltas. This is the only
///   type that carries the time series.
/// * `2` — a periodic metadata delta. MongoDB 7 and later write these for
///   slow-changing values such as server parameters. The payload is `doc`
///   plus a `count`. A metrics file usually has one of these *before* the
///   first metric chunk. They are not samples and they are not a reason to
///   stop reading.
///
/// The server has grown this set before (`2` itself was the addition). A
/// type this build does not know is skipped and remembered, so metric chunks
/// that follow it are still decoded.
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
    metadata_docs: Vec<Document>,
    /// Type 2 `doc` sub-documents, in file order.
    ///
    /// The old field name for this was `parameters`. Type 2 is a metadata
    /// delta, not the startup parameter block. Several of them can appear
    /// in one file; keeping only the last one drops the earlier samples.
    metadata_deltas: Vec<Document>,
    /// BSON documents visited while reading, of every type, including
    /// types that were skipped.
    document_count: usize,
    /// `type` numbers this build did not recognise. Reading continued.
    skipped_chunk_types: Vec<i32>,
    /// Type 1 `data` blobs, still compressed, in file order.
    binary_chunks: Vec<Binary>,
}

#[derive(Debug)]
struct MetricData {
    name: String,
    initial: u64,
    /// BSON value this sample was taken from. Kept so a metric_count
    /// mismatch can say which field disagreed with the compressor.
    #[allow(dead_code)]
    metric_type: Bson,
    metrics: Option<usize>,
    /// Reconstructed samples for one chunk.
    ///
    /// Index 0 is the value taken from the reference document. Each later
    /// index is that value plus the deltas that follow it. The length is
    /// `sample_count + 1`, not `sample_count`: the header counts deltas,
    /// and the reference document is itself a sample.
    data: Option<Vec<u64>>,
}

struct RangeCursor {
    buffer: Vec<u8>,
    _size: usize,
    at: usize,
}

/// Read one unsigned LEB128 integer from the delta block.
///
/// FTDC uses the same 7-bit grouping as the S2 varint: the low 7 bits
/// are payload, and the high bit says another byte follows. A zero is a
/// real value (it starts a run of zeros). Running off the end of the
/// buffer used to return 0 as well, so a short chunk was decoded as a
/// run of quiet zeros instead of a broken file.
fn unpack(data_range_cursor: &mut RangeCursor) -> Result<u64, FileDecoderError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    loop {
        if data_range_cursor.at >= data_range_cursor.buffer.len() {
            return Err(FileDecoderError::Other(anyhow!(
                "FTDC varint ran past the end of the delta block"
            )));
        }
        // A u64 has room for 10 payload bytes (9 * 7 + 1). One more byte
        // would shift by 70 and panic, and it cannot be a legal value.
        if shift >= 64 {
            return Err(FileDecoderError::Other(anyhow!(
                "FTDC varint is wider than 64 bits"
            )));
        }
        let b = data_range_cursor.buffer[data_range_cursor.at];
        data_range_cursor.at += 1;

        result |= u64::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

/// Inflate one metric blob.
///
/// The first four bytes are the uncompressed size the server writes in
/// front of the zlib stream (`metrics_chunk.uncompressed_size`). They
/// are not zlib bytes. A blob shorter than that is not a metric chunk.
fn inflate_owned(bytes: Vec<u8>) -> Result<Vec<u8>, FileDecoderError> {
    if bytes.len() < 4 {
        return Err(FileDecoderError::Other(anyhow!(
            "metric blob is shorter than the 4-byte uncompressed size"
        )));
    }
    let mut decoder = DeflateDecoder::new(&bytes[4..]);
    match decoder.decode_zlib() {
        Ok(encoded_data) => Ok(encoded_data),
        Err(error) => Err(FileDecoderError::Other(anyhow!(
            "zlib inflate of an FTDC metric chunk failed: {error}"
        ))),
    }
}

/// Reconstruct one inflated chunk as `(name, samples)` pairs, in walk order.
///
/// The name still has its leading dot stripped here. Publishing the path
/// without that dot is a separate change. This function does not append
/// to a shared map: the caller appends, in chunk order.
fn decode_one_chunk(chunk: Vec<u8>) -> Result<Vec<(String, Vec<u64>)>, FileDecoderError> {
    let mut metric_data = FTDCFileDecoder::extract_metric_meta_data(&chunk)?;
    FTDCFileDecoder::extract_all_metrics_from_buffer(&mut metric_data, &chunk)?;

    let mut columns = Vec::with_capacity(metric_data.len());
    for metric in metric_data {
        let Some(samples) = metric.data else {
            return Err(FileDecoderError::Other(anyhow!(
                "metric '{}' produced no samples",
                metric.name
            )));
        };
        let mut name = metric.name;
        // The walk prefixes every segment with a dot, including the first.
        if name.starts_with('.') {
            name.remove(0);
        }
        columns.push((name, samples));
    }
    Ok(columns)
}

fn push_metric(name: &str, initial: u64, metric_type: Bson, out: &mut Vec<MetricData>) {
    out.push(MetricData {
        name: name.to_string(),
        initial,
        metric_type,
        metrics: None,
        data: None,
    });
}

/// Convert a Decimal128 the way `BSONElement::numberLong()` does.
///
/// The compressor does not have a special case for decimal. It calls
/// `numberLong()`, and `Decimal128::toLong()` truncates toward zero and
/// saturates at the int64 limits. NaN becomes 0. The int64 bit pattern
/// is then stored as the uint64 metric, so a negative value is a large
/// unsigned number, which is what a later wrapping delta expects.
///
/// The `bson` crate exposes Decimal128 as text (`"42.9"`, `"1.05E+3"`,
/// `"NaN"`, `"Infinity"`). The integer is taken from that text.
fn decimal128_to_metric(value: &bson::Decimal128) -> u64 {
    decimal_text_to_i64(&value.to_string()) as u64
}

fn decimal_text_to_i64(text: &str) -> i64 {
    match text {
        "NaN" | "nan" | "NAN" => return 0,
        "Infinity" | "+Infinity" => return i64::MAX,
        "-Infinity" => return i64::MIN,
        _ => {}
    }

    let negative = text.starts_with('-');
    let body = text.trim_start_matches(['+', '-']);
    let (coeff, exponent) = match body.split_once(['e', 'E']) {
        Some((coeff, exp)) => match exp.parse::<i32>() {
            Ok(exp) => (coeff, exp),
            Err(_) => return 0,
        },
        None => (body, 0),
    };
    let (int_digits, frac_digits) = match coeff.split_once('.') {
        Some((int_digits, frac_digits)) => (int_digits, frac_digits),
        None => (coeff, ""),
    };
    if int_digits.is_empty()
        || !int_digits.bytes().all(|b| b.is_ascii_digit())
        || !frac_digits.bytes().all(|b| b.is_ascii_digit())
    {
        return 0;
    }

    let digits = format!("{int_digits}{frac_digits}");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return 0;
    }

    // scale is the power of ten that turns the digit string into the
    // integer part. A negative scale means "drop digits", which is the
    // truncation toward zero.
    let scale = exponent - frac_digits.len() as i32;
    let mut magnitude = digits.to_string();
    if scale >= 0 {
        magnitude.extend(std::iter::repeat('0').take(scale as usize));
    } else {
        let drop = (-scale) as usize;
        if drop >= magnitude.len() {
            return 0;
        }
        magnitude.truncate(magnitude.len() - drop);
    }

    let parsed = match magnitude.parse::<u64>() {
        Ok(value) => value,
        // More digits than a u64. The int64 range is smaller still.
        Err(_) => return if negative { i64::MIN } else { i64::MAX },
    };
    if !negative {
        if parsed > i64::MAX as u64 {
            i64::MAX
        } else {
            parsed as i64
        }
    } else if parsed > i64::MAX as u64 {
        i64::MIN
    } else {
        -(parsed as i64)
    }
}

impl FTDCFileDecoder {
    pub const INITIAL_METRIC_CAPACITY: usize = 10240;

    /// The server refuses a chunk whose `metric_count * sample_count`
    /// exceeds this. A corrupt header that asks for more than that is
    /// rejected here so we do not allocate it.
    const MAX_METRIC_SAMPLES: usize = 1_000_000;

    /// Read the three words that precede the delta block.
    ///
    /// Layout of an inflated metric chunk, after the 4-byte uncompressed
    /// size and the zlib wrapper have been removed:
    ///
    /// ```text
    /// reference document   BSON, whose own first 4 bytes are its length
    /// metric_count         uint32   how many numbers were extracted
    /// sample_count         uint32   how many *deltas* follow, per metric
    /// deltas               varints
    /// ```
    ///
    /// `sample_count` does not include the reference document. The server
    /// decompressor returns `1 + sample_count` samples: the reference,
    /// then one reconstructed document per delta. A chunk that only holds
    /// the reference (common at the start of a file) has `sample_count`
    /// 0 and still has one sample.
    fn read_header(data: &[u8]) -> Result<(usize, usize, usize), FileDecoderError> {
        if data.len() < 4 {
            return Err(FileDecoderError::Other(anyhow!(
                "inflated metric chunk is shorter than a BSON length"
            )));
        }
        let data_size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        if data_size < 5 || data.len() < data_size.saturating_add(8) {
            return Err(FileDecoderError::Other(anyhow!(
                "inflated metric chunk does not contain metric_count and sample_count"
            )));
        }
        let metric_count =
            u32::from_le_bytes(data[data_size..data_size + 4].try_into().unwrap()) as usize;
        let sample_count =
            u32::from_le_bytes(data[data_size + 4..data_size + 8].try_into().unwrap()) as usize;
        if metric_count.saturating_mul(sample_count) > Self::MAX_METRIC_SAMPLES {
            return Err(FileDecoderError::Other(anyhow!(
                "metric_count {metric_count} and sample_count {sample_count} exceed the FTDC limit"
            )));
        }
        Ok((data_size, metric_count, sample_count))
    }

    /// Inflate every metric chunk, and return the chunks in file order.
    ///
    /// Inflation has no cross-chunk dependency, so each blob runs on its
    /// own thread. The output does. Sample 0 of chunk N+1 is the sample
    /// after the last sample of chunk N. Pushing into a shared `Vec`
    /// from whichever thread took the lock first scrambled that order,
    /// and it scrambled it differently on every run.
    ///
    /// The handles are joined in the order they were spawned. A later
    /// chunk finishing first just waits its turn. The `Result` from the
    /// worker is returned; it used to be discarded with `let _ =`, so a
    /// chunk that failed to inflate disappeared and the file still
    /// looked successful.
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
                .map_err(|_| FileDecoderError::Other(anyhow!("FTDC inflate thread panicked")))??;
            inflated.push(chunk);
        }
        Ok(inflated)
    }

    /// Walk a reference document in field order and record one metric per
    /// number the server compressor would have extracted.
    ///
    /// `FTDCBSONUtil::isFTDCType` is the list. Everything else is skipped,
    /// not turned into a number:
    ///
    /// * `double`, `int32`, `int64`, `decimal128`, `bool`, `datetime` — one
    ///   `uint64` each
    /// * `timestamp` — two `uint64`s, seconds then increment
    /// * subdocuments and arrays — walked, not themselves a metric
    ///
    /// Strings, ObjectIds, binary, null, and the other BSON types are not
    /// on that list. `serverStatus.repl.topologyVersion.processId` and
    /// `electionId` are ObjectIds. Counting either of them makes this walk
    /// longer than `metric_count` in the chunk header, and every delta
    /// after that point is applied to the wrong series.
    ///
    /// An array is a BSON document whose keys are `"0"`, `"1"`, ... so the
    /// path of an element is `parent.key.<index>`, the same way a nested
    /// document appends its field name. Numeric elements of an array are
    /// metrics. Skipping them (and only descending into subdocuments)
    /// shortens the walk by the same amount.
    ///
    /// `doc_name` is the path so far, with a leading dot on every segment
    /// including the first. Callers strip that first character when they
    /// publish the name. The dot is a historical accident of starting from
    /// an empty parent; the path cleanup removes it.
    fn extract_metric_data_from_doc(
        doc_name: &str,
        root: Document,
        metric_data: &mut Vec<MetricData>,
    ) {
        for (key, value) in root {
            let path = format!("{doc_name}.{key}");
            match value {
                Bson::Double(v) => {
                    // Range reduction to int64 is applied in a later change.
                    // `as u64` is not what the compressor stored for NaN,
                    // negatives, or values past 2^63.
                    push_metric(&path, v as u64, value, metric_data);
                }
                Bson::Int32(v) => push_metric(&path, v as u64, value, metric_data),
                Bson::Int64(v) => push_metric(&path, v as u64, value, metric_data),
                Bson::Decimal128(v) => {
                    // `BSONElement::numberLong()` for NumberDecimal, which is
                    // what the compressor calls. One uint64, not a sentinel.
                    // The sentinel 999999999 used to be the base every delta
                    // in this column was added to.
                    push_metric(&path, decimal128_to_metric(&v), value, metric_data);
                }
                Bson::Boolean(v) => push_metric(&path, u64::from(v), value, metric_data),
                Bson::DateTime(v) => {
                    push_metric(&path, v.timestamp_millis() as u64, value, metric_data);
                }
                Bson::Timestamp(v) => {
                    // Two metrics, seconds then increment. The names gain a
                    // suffix so the two columns can be told apart; the order
                    // is the order the compressor pushes them.
                    push_metric(
                        &format!("{path}_time"),
                        u64::from(v.time),
                        value.clone(),
                        metric_data,
                    );
                    push_metric(
                        &format!("{path}_increment"),
                        u64::from(v.increment),
                        value,
                        metric_data,
                    );
                }
                Bson::Document(sub_doc) => {
                    FTDCFileDecoder::extract_metric_data_from_doc(&path, sub_doc, metric_data);
                }
                Bson::Array(items) => {
                    for (index, item) in items.into_iter().enumerate() {
                        // Re-enter through the same match by wrapping the
                        // element in a one-field document whose key is the
                        // index. That keeps array elements and document
                        // fields on one path.
                        let mut wrapper = Document::new();
                        wrapper.insert(index.to_string(), item);
                        FTDCFileDecoder::extract_metric_data_from_doc(&path, wrapper, metric_data);
                    }
                }
                // String, ObjectId, Null, Binary, Regex, JavaScript, Symbol,
                // Undefined, MinKey, MaxKey, DBPointer. Not metrics.
                _ => {}
            }
        }
    }

    /// Extracts the names of the metrics and their count, as the data in the header is not reliable*
    ///
    /// # Arguments
    ///
    /// *  `buffer` - The decompressed data block, that is cast to d BSON Document.
    ///
    /// # Returns
    ///
    /// * A vector of `MetricData`
    ///
    fn extract_metric_meta_data(buffer: &Vec<u8>) -> Result<Vec<MetricData>, FileDecoderError> {
        let mut base = buffer.as_slice();

        match Document::from_reader(&mut base) {
            Ok(bson_doc) => {
                let mut ret_metrics: Vec<MetricData> = vec![];

                let root = bson_doc;

                FTDCFileDecoder::extract_metric_data_from_doc("", root, &mut ret_metrics);

                Ok(ret_metrics)
            }
            Err(e) => Err(FileDecoderError::Other(anyhow!(e))),
        }
    }

    /// Rebuild every metric in one inflated chunk.
    ///
    /// The delta block is column-major: every delta of metric 0, then
    /// every delta of metric 1, and so on. A delta is added to the
    /// previous sample with wrapping `uint64` arithmetic, which is what
    /// the C++ compressor does (`uint64_t` addition). The reference
    /// value is sample 0 and is not itself stored as a delta.
    ///
    /// Zeros are run-length encoded across that whole block, including
    /// across the boundary between two metrics. A run of *k* zeros is
    /// stored as the pair `[0, k - 1]`: the `0` is the first zero, and
    /// the next varint is how many further zeros follow. The count is
    /// shared across metrics on purpose. Resetting it at each metric
    /// would swallow the run that the compressor wrote as one pair.
    ///
    /// The returned count is the number of reconstructed sample values
    /// (`metrics * (sample_count + 1)`). Each `MetricData::data` vector
    /// has length `sample_count + 1`.
    fn extract_all_metrics_from_buffer(
        metric_data_vec: &mut Vec<MetricData>,
        buffer: &Vec<u8>,
    ) -> Result<usize, FileDecoderError> {
        let (doc_size, metric_count, delta_count) =
            FTDCFileDecoder::read_header(buffer.as_slice())?;

        // The delta block has no names. Column i of the block belongs to
        // metric i of this walk. The server checks the same equality and
        // refuses the chunk when it fails. Doing it here, before any
        // addition, means a bad walk cannot silently shift the series.
        if metric_data_vec.len() != metric_count {
            let first = metric_data_vec
                .first()
                .map(|metric| metric.name.as_str())
                .unwrap_or("(none)");
            let last = metric_data_vec
                .last()
                .map(|metric| metric.name.as_str())
                .unwrap_or("(none)");
            return Err(FileDecoderError::Other(anyhow!(
                "extracted {} metrics from the reference document, but the chunk header says {metric_count}. \
                 Deltas are positional, so these lengths have to match or every later series is built from the wrong column. \
                 First extracted name is '{first}', last is '{last}'.",
                metric_data_vec.len()
            )));
        }

        let offset = doc_size + 8;
        if offset > buffer.len() {
            return Err(FileDecoderError::Other(anyhow!(
                "metric chunk header points past the end of the inflated buffer"
            )));
        }
        let mut data_range_cursor = RangeCursor {
            buffer: buffer[offset..].to_vec(),
            _size: buffer.len() - offset,
            at: 0,
        };

        let mut count = 0;
        // Additional zeros still to emit. This survives from one metric
        // to the next. See the comment on this function.
        let mut zeros_remaining: u64 = 0;
        for metric in metric_data_vec.iter_mut() {
            let mut value = metric.initial;
            let mut samples = Vec::with_capacity(delta_count + 1);
            samples.push(value);

            for _ in 0..delta_count {
                let delta = if zeros_remaining != 0 {
                    zeros_remaining -= 1;
                    0
                } else {
                    let decoded = unpack(&mut data_range_cursor)?;
                    if decoded == 0 {
                        zeros_remaining = unpack(&mut data_range_cursor)?;
                    }
                    decoded
                };

                // Wrapping add. A delta of u64::MAX takes a counter
                // backwards by one, which is how a value that decreased
                // is encoded. Checked add would fail those samples.
                value = value.wrapping_add(delta);
                samples.push(value);
                count += 1;
            }

            // Length includes the reference sample. Callers that kept
            // only `delta_count` elements dropped the last sample of
            // every chunk, and dropped the only sample of a chunk whose
            // sample_count was 0.
            metric.metrics = Some(samples.len());
            metric.data = Some(samples);
            count += 1;
        }

        Ok(count)
    }

    /// Creates a new structure to decode a single file
    ///
    /// # Arguments
    ///
    /// # Returns
    /// A clean `FTDCFileDecoder` structure.
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
    ///
    /// # Returns
    ///
    /// * An empty ()
    ///
    pub fn read_file(&mut self, file_name: &str) -> Result<(), FileDecoderError> {
        // Open file, get buffed_reader
        match buffered_reader::File::open(file_name) {
            Ok(mut reader) => {
                loop {
                    // An FTDC file is a series of BSON documents.
                    // Read them in one by one
                    match Document::from_reader(&mut reader) {
                        // If it can be read...
                        Ok(metrics) => {
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
                        Err(e) => {
                            if !reader.eof() {
                                return Err(FileDecoderError::Other(anyhow!(e)));
                            }
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                // IO errors
                return Err(FileDecoderError::FileIOError(e));
            }
        }

        Ok(())
    }

    /// Reads an FTDC file and decompresses the data bearing documents in it.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file to read.
    ///
    /// # Returns
    ///
    /// * A vector of vectors of U8 values.
    ///
    pub fn read_and_uncompress(
        &mut self,
        file_name: &str,
    ) -> Result<Vec<Vec<u8>>, FileDecoderError> {
        match self.read_file(file_name) {
            Ok(_) => self.uncompress_data(),
            Err(e) => {
                println!("Got expected error {e:?}");
                Err(e)
            }
        }
    }

    /// Decode inflated chunks and append each metric in file order.
    ///
    /// Each chunk is decoded on its own thread, because the reference
    /// walk and the delta reconstruction do not look at other chunks.
    /// The append does. A metric's samples are the concatenation of its
    /// samples from chunk 0, then chunk 1, then chunk 2. Joining the
    /// workers in spawn order and appending only after the previous
    /// chunk has been appended is what keeps that concatenation in time
    /// order. Appending from inside the worker, under a mutex, appended
    /// in completion order instead.
    ///
    /// A worker's `Err` stops the decode. Previously the `Result` was
    /// ignored, so a chunk whose `metric_count` did not match the
    /// reference walk was dropped and the rest of the file was returned
    /// as success.
    pub fn decode_decompressed_data_chunks(
        &self,
        data_chunks: Vec<Vec<u8>>,
    ) -> Result<HashMap<String, Vec<u64>>, FileDecoderError> {
        let mut handles = Vec::with_capacity(data_chunks.len());
        for chunk in data_chunks {
            handles.push(thread::spawn(move || decode_one_chunk(chunk)));
        }

        let mut merged: HashMap<String, Vec<u64>> = HashMap::new();
        for handle in handles {
            let metrics = handle
                .join()
                .map_err(|_| FileDecoderError::Other(anyhow!("FTDC decode thread panicked")))??;
            for (name, mut samples) in metrics {
                merged
                    .entry(name)
                    .or_insert_with(|| Vec::with_capacity(Self::INITIAL_METRIC_CAPACITY))
                    .append(&mut samples);
            }
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
    /// * A hashmap of the metric name and a vector of U64 values.
    ///
    pub fn read_and_decode(
        &mut self,
        file_name: &str,
        filter: Option<Regex>,
    ) -> Result<HashMap<String, Vec<u64>>, FileDecoderError> {
        let data_chunks = self.read_and_uncompress(file_name)?;
        let decoded_data = self.decode_decompressed_data_chunks(data_chunks)?;

        match filter {
            Some(re) => {
                let mut matched_data: HashMap<String, Vec<u64>> = HashMap::new();

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

    /// Converts a HashMap<String, Vec<u64>> to a polars DataFrame
    ///
    /// # Arguments
    /// * `metrics` - A HashMap where keys are metric names and values are vectors of u64 values
    ///
    /// # Returns
    /// * `Result<DataFrame, FileDecoderError>` - A polars DataFrame with columns named after the keys in the hashmap
    pub fn metrics_to_dataframe(
        metrics: HashMap<String, Vec<u64>>,
    ) -> Result<DataFrame, FileDecoderError> {
        // Create a vector to hold our Series objects
        let mut series_vec: Vec<Series> = Vec::new();

        // Find the maximum length of any vector in the hashmap to determine row count
        let max_len = metrics.values().map(|v| v.len()).max().unwrap_or(0);

        // Create a Series for each key-value pair in the hashmap
        for (key, values) in metrics {
            // If the vector is shorter than max_len, pad it with zeros
            let mut padded_values = values.clone();
            if padded_values.len() < max_len {
                padded_values.resize(max_len, 0);
            }

            // Create a Series from the vector and add it to our vector of Series
            // Convert the key to PlSmallStr using .into() without the &
            let series = Series::new(key.into(), padded_values);
            series_vec.push(series);
        }

        // Create a DataFrame from the vector of Series
        // DataFrame::from_iter returns a DataFrame directly, not a Result
        Ok(DataFrame::from_iter(series_vec))
    }

    /// Decodes an FTDC file and writes a Parquet file with the information.
    ///
    /// # Arguments
    ///
    /// * `input_file_name` - The name of the FTDC file to decode.
    /// * `output_file_name` - The name of the file to write parquet data.
    /// * `metadata` - An Option<&str> that if not None will be used to write metadata.
    /// # Returns
    ///
    /// * `Result<(), FileDecoderError>` - Nothing
    pub fn write_to_parquet(
        input_file_name: &str,
        output_file_name: &str,
        metadata: Option<&str>,
        filter: Option<Regex>,
    ) -> Result<(), FileDecoderError> {
        let mut parsed_file = FTDCFileDecoder::new();

        let metrics = parsed_file.read_and_decode(input_file_name, filter)?;

        // Convert the metrics hashmap to a DataFrame
        let mut df = Self::metrics_to_dataframe(metrics)?;

        // Write the DataFrame to a parquet file
        let mut file = std::fs::File::create(output_file_name)
            .map_err(|e| FileDecoderError::Other(anyhow!("Failed to create output file: {}", e)))?;

        ParquetWriter::new(&mut file)
            .finish(&mut df)
            .map_err(|e| FileDecoderError::Other(anyhow!("Failed to write parquet file: {}", e)))?;

        // Write metadata
        if let Some(metadata_file) = metadata {
            // Create entry
            let metric_names: Vec<String> = df
                .get_column_names()
                .iter()
                .map(move |name| -> String { name.to_string() })
                .collect();

            // Convert the Metadata struct to a JSON string
            // The sidecar still stores one string per group. The reader
            // keeps every document; the export that writes them as a JSON
            // array lands with the CLI changes. `Document`'s `Display` is
            // what the previous field stored, so the same text is kept here.
            let metadata = parsed_file
                .metadata_docs
                .last()
                .map(|doc| doc.to_string())
                .unwrap_or_default();
            let parameters = parsed_file
                .metadata_deltas
                .last()
                .map(|doc| doc.to_string())
                .unwrap_or_default();
            let json_data = serde_json::json!({
                "metadata": metadata,
                "parameters": parameters,
                "chunks_in_file": parsed_file.document_count,
                "metrics": metric_names,

            });

            // Write the JSON string to the specified file
            let json_string = serde_json::to_string_pretty(&json_data).map_err(|e| {
                FileDecoderError::Other(anyhow!("Failed to serialize metadata to JSON: {}", e))
            })?;

            std::fs::write(&metadata_file, json_string).map_err(|e| {
                FileDecoderError::Other(anyhow!("Failed to write metadata file: {}", e))
            })?;
        }

        Ok(())
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
            FileDecoderError::Other(anyhow!("FTDC chunk type {value} does not fit in an int32"))
        }),
        Some(other) => Err(FileDecoderError::Other(anyhow!(
            "FTDC chunk type is {other:?}, expected an int"
        ))),
        None => Err(FileDecoderError::Other(anyhow!(
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
        Some(other) => Err(FileDecoderError::Other(anyhow!(
            "'{key}' in an FTDC {kind} chunk is {other:?}, expected a document"
        ))),
        None => Err(FileDecoderError::Other(anyhow!(
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
        Some(other) => Err(FileDecoderError::Other(anyhow!(
            "'data' in an FTDC metric chunk is {other:?}, expected BinData"
        ))),
        None => Err(FileDecoderError::Other(anyhow!(
            "'data' key not found in an FTDC metric chunk"
        ))),
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#[cfg(test)]

mod tests {
    use crate::FTDCFileDecoder;
    use std::collections::HashMap;

    //const TEST_FILE_PATH: &str = "test_data/diagnostic.data/metrics.2025-04-15T16-48-31Z-00000";

    const TEST_FILE_PATH: &str =
        "./test_data/diagnostic.data.8.0.11/metrics.2025-07-23T20-33-08Z-00000";

    #[test]
    fn test_file_doesnt_exist() {
        let mut bad_file = FTDCFileDecoder::new();
        let result = bad_file.read_file("unexistent");
        match result {
            Ok(_read) => {
                assert!(false)
            }
            Err(e) => {
                println!("Got expected error {e:?}");
            }
        }
    }

    #[test]
    #[ignore = "fixture diagnostic.data.8.0.11 is not in the repository"]
    fn test_should_read_5_chunks() {
        let mut parsed_file = FTDCFileDecoder::new();
        parsed_file.read_file(TEST_FILE_PATH).unwrap();
        assert_eq!(parsed_file.document_count, 213);
    }

    /// A MongoDB 7+ archive starts with metadata, then a type 2 delta,
    /// and only then the metric chunks. An unknown type must not end
    /// the read, and a second delta must not replace the first.
    #[test]
    fn type2_and_unknown_types_do_not_abort_the_file() {
        let path =
            std::env::temp_dir().join(format!("ftdc-rs-chunk-types-{}.bin", std::process::id()));
        let mut bytes = Vec::new();
        let documents = [
            bson::doc! {
                "_id": bson::DateTime::from_millis(0),
                "type": 0i32,
                "doc": { "host": "a.example" }
            },
            bson::doc! {
                "_id": bson::DateTime::from_millis(1),
                "type": 2i32,
                "count": 0i32,
                "doc": { "param": 1i32 }
            },
            bson::doc! {
                "_id": bson::DateTime::from_millis(2),
                "type": 9i32,
                "doc": { "future": true }
            },
            bson::doc! {
                "_id": bson::DateTime::from_millis(3),
                "type": 2i32,
                "count": 1i32,
                "doc": { "param": 2i32 }
            },
        ];
        for document in &documents {
            document.to_writer(&mut bytes).unwrap();
        }
        std::fs::write(&path, &bytes).unwrap();

        let mut parsed_file = FTDCFileDecoder::new();
        parsed_file
            .read_file(path.to_str().unwrap())
            .expect("type 2 and an unknown type are not fatal");

        assert_eq!(parsed_file.metadata_docs.len(), 1);
        assert_eq!(
            parsed_file.metadata_docs[0].get_str("host").unwrap(),
            "a.example"
        );
        assert_eq!(parsed_file.metadata_deltas.len(), 2);
        assert_eq!(parsed_file.metadata_deltas[0].get_i32("param").unwrap(), 1);
        assert_eq!(parsed_file.metadata_deltas[1].get_i32("param").unwrap(), 2);
        assert_eq!(parsed_file.skipped_chunk_types, vec![9]);
        assert_eq!(parsed_file.document_count, 4);
        assert!(parsed_file.binary_chunks.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "fixture diagnostic.data.8.0.11 is not in the repository"]
    fn test_inflate_chunks() {
        let mut parsed_file = FTDCFileDecoder::new();
        parsed_file.read_file(TEST_FILE_PATH).unwrap();

        let uncompressed_vec = parsed_file.uncompress_data();

        assert!(uncompressed_vec.is_ok());

        match uncompressed_vec {
            Ok(uncompressed_vec) => {
                assert_eq!(uncompressed_vec.len(), parsed_file.binary_chunks.len());
                assert_eq!(uncompressed_vec.len(), 211);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    #[ignore = "fixture diagnostic.data.8.0.11 is not in the repository"]
    fn get_metric_meta_data() -> () {
        let mut parsed_file = FTDCFileDecoder::new();
        parsed_file.read_file(TEST_FILE_PATH).unwrap();

        let uncompressed_vec = parsed_file.uncompress_data();

        assert!(uncompressed_vec.is_ok());

        let all_chunks = uncompressed_vec.unwrap();

        for chunk in &all_chunks {
            match FTDCFileDecoder::extract_metric_meta_data(&chunk) {
                Ok(metrics) => {
                    println!("{:?}", metrics);
                }
                Err(_e) => {
                    assert!(false);
                }
            }
        }

        assert!(true);
    }

    #[test]
    #[ignore = "fixture diagnostic.data.8.0.11 is not in the repository"]
    fn test_metric_deltas() -> () {
        let mut parsed_file = FTDCFileDecoder::new();
        parsed_file.read_file(TEST_FILE_PATH).unwrap();

        let uncompressed_vec = parsed_file.uncompress_data();

        assert!(uncompressed_vec.is_ok());
        let all_chunks = uncompressed_vec.unwrap();

        let mut chunk_number = 0;
        for chunk in &all_chunks {
            match FTDCFileDecoder::extract_metric_meta_data(&chunk) {
                Ok(mut metrics) => {
                    match FTDCFileDecoder::extract_all_metrics_from_buffer(&mut metrics, chunk) {
                        Ok(metric_count) => {
                            println!(
                                "chunk {chunk_number} {metric_count} = {}",
                                metrics.get(0).unwrap().metrics.unwrap()
                            );
                        }
                        Err(e) => {
                            println!("Chunk number {chunk_number}\nGot error {e:?}");
                            assert!(false);
                        }
                    }
                }
                Err(_e) => {
                    assert!(false);
                }
            }
            chunk_number += 1;
        }

        assert!(true);
    }

    #[test]
    #[ignore = "fixture diagnostic.data.8.0.11 is not in the repository"]
    fn test_get_metric_data_decoded() -> () {
        let mut parsed_file = FTDCFileDecoder::new();
        let data_vec = parsed_file.read_and_decode(TEST_FILE_PATH, None).unwrap();

        data_vec.iter().for_each(|x| {
            println!("{}", x.0);
        });

        assert!(true);
    }

    /// Encode `n` as the unsigned LEB128 that [`unpack`] reads back.
    fn encode_varint(mut n: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (n & 0x7f) as u8;
            n >>= 7;
            if n != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if n == 0 {
                break;
            }
        }
        out
    }

    /// Encode a column-major delta list the way the compressor does.
    ///
    /// A run of *k* zeros becomes the two varints `[0, k - 1]`. The
    /// first 0 is a zero delta. The second number is how many more
    /// zeros follow it, not how many zeros there were in total.
    fn encode_deltas(deltas: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut index = 0;
        while index < deltas.len() {
            if deltas[index] != 0 {
                out.extend(encode_varint(deltas[index]));
                index += 1;
                continue;
            }
            let start = index;
            while index < deltas.len() && deltas[index] == 0 {
                index += 1;
            }
            let run = (index - start) as u64;
            out.extend(encode_varint(0));
            out.extend(encode_varint(run - 1));
        }
        out
    }

    /// Build a one-chunk FTDC file and return its path.
    ///
    /// `sample_count` is the header value: the number of deltas per
    /// metric, not the number of samples. `deltas` is column-major,
    /// metric 0's deltas first.
    fn write_one_chunk(
        reference: &bson::Document,
        metric_count: u32,
        sample_count: u32,
        deltas: &[u64],
    ) -> std::path::PathBuf {
        use std::io::Write;

        let mut payload = Vec::new();
        reference.to_writer(&mut payload).unwrap();
        payload.extend_from_slice(&metric_count.to_le_bytes());
        payload.extend_from_slice(&sample_count.to_le_bytes());
        payload.extend(encode_deltas(deltas));

        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&payload).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut blob = Vec::with_capacity(4 + compressed.len());
        blob.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        blob.extend(compressed);

        let wrapper = bson::doc! {
            "_id": bson::DateTime::from_millis(0),
            "type": 1i32,
            "data": bson::Binary {
                subtype: bson::spec::BinarySubtype::Generic,
                bytes: blob,
            }
        };
        let mut file = Vec::new();
        wrapper.to_writer(&mut file).unwrap();

        // Unique per call. Several tests build a chunk with the same
        // sample_count, and `cargo test` runs them on more than one thread.
        use std::sync::atomic::{AtomicU64, Ordering};
        static CHUNK_SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ftdc-rs-chunk-{}-{}.bin",
            std::process::id(),
            CHUNK_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, file).unwrap();
        path
    }

    /// The walkthrough in the server FTDC README, plus the two edges the
    /// fixed `[u64; 300]` buffer got wrong.
    ///
    /// Four documents `{a, x, s}` become a reference of `{a:1, x:2}`
    /// (the string is not a metric) and the delta block `[1, 1, 1, 0, 2]`.
    /// The reconstructed series have four points, not three. A chunk
    /// with `sample_count == 0` still yields the reference. A chunk with
    /// 300 deltas used to write `metric_values[300]` and panic.
    #[test]
    fn reference_sample_is_kept_and_a_full_chunk_does_not_panic() {
        let reference = bson::doc! { "a": 1i32, "x": 2i32, "s": "t" };
        // a grows 1,2,3,4. x stays 2. Three zeros encode as [0, 2].
        let path = write_one_chunk(&reference, 2, 3, &[1, 1, 1, 0, 0, 0]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(
            series.get("a").map(Vec::as_slice),
            Some([1, 2, 3, 4].as_slice())
        );
        assert_eq!(
            series.get("x").map(Vec::as_slice),
            Some([2, 2, 2, 2].as_slice())
        );
        let _ = std::fs::remove_file(&path);

        let reference_only = write_one_chunk(&reference, 2, 0, &[]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(reference_only.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(series.get("a").map(Vec::as_slice), Some([1].as_slice()));
        assert_eq!(series.get("x").map(Vec::as_slice), Some([2].as_slice()));
        let _ = std::fs::remove_file(&reference_only);

        let wide = bson::doc! { "n": 0i64 };
        let path = write_one_chunk(&wide, 1, 300, &vec![1u64; 300]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        let values = series.get("n").expect("metric n");
        assert_eq!(values.len(), 301);
        assert_eq!(values[0], 0);
        assert_eq!(values[300], 300);
        let _ = std::fs::remove_file(&path);
    }

    /// ObjectId and string are not metrics. Decimal128 is one int64,
    /// truncated toward zero. An array element is `key.<index>`.
    /// A header count that does not match the walk is an error.
    #[test]
    fn extraction_matches_the_compressor_metric_list() {
        let oid = bson::oid::ObjectId::from_bytes([0x11; 12]);
        let reference = bson::doc! {
            "label": "host",
            "id": oid,
            "n": 7i32,
            "ok": true,
        };
        let path = write_one_chunk(&reference, 2, 0, &[]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(series.len(), 2, "objectid and string must not be columns");
        assert_eq!(series.get("n").map(Vec::as_slice), Some([7].as_slice()));
        assert_eq!(series.get("ok").map(Vec::as_slice), Some([1].as_slice()));
        assert!(!series.contains_key("id"));
        assert!(!series.contains_key("label"));
        let _ = std::fs::remove_file(&path);

        // Counting the ObjectId would make the walk 3 long. The header
        // says 3 on purpose, which is what a decoder that emits ObjectIds
        // would believe. The lengths disagree, so the delta block must
        // not be applied. The check is on the chunk itself; the threaded
        // merge still drops worker errors, and that is fixed separately.
        let path = write_one_chunk(&reference, 3, 0, &[]);
        let mut parsed = FTDCFileDecoder::new();
        parsed.read_file(path.to_str().unwrap()).unwrap();
        let inflated = parsed.uncompress_data().unwrap();
        let mut metrics = FTDCFileDecoder::extract_metric_meta_data(&inflated[0]).unwrap();
        let err = FTDCFileDecoder::extract_all_metrics_from_buffer(&mut metrics, &inflated[0])
            .expect_err("a walk that does not match metric_count must fail");
        let message = err.to_string();
        assert!(message.contains("3"), "{message}");
        let _ = std::fs::remove_file(&path);

        let decimal: bson::Decimal128 = "42.9".parse().unwrap();
        let reference = bson::doc! { "amount": decimal };
        let path = write_one_chunk(&reference, 1, 0, &[]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(
            series.get("amount").map(Vec::as_slice),
            Some([42].as_slice())
        );
        let _ = std::fs::remove_file(&path);

        let reference = bson::doc! {
            "members": [ { "health": 1i32 }, { "health": 2i32 } ]
        };
        let path = write_one_chunk(&reference, 2, 0, &[]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(
            series.get("members.0.health").map(Vec::as_slice),
            Some([1].as_slice())
        );
        assert_eq!(
            series.get("members.1.health").map(Vec::as_slice),
            Some([2].as_slice())
        );
        let _ = std::fs::remove_file(&path);

        assert_eq!(super::decimal_text_to_i64("1.05E+3"), 1050);
        assert_eq!(super::decimal_text_to_i64("-3.2"), -3);
        assert_eq!(super::decimal_text_to_i64("NaN"), 0);
        assert_eq!(super::decimal_text_to_i64("Infinity"), i64::MAX);
        assert_eq!(super::decimal_text_to_i64("-Infinity"), i64::MIN);
    }

    /// Two chunks must concatenate in file order, and a chunk whose
    /// metric_count does not match the walk must fail the whole decode.
    /// Both used to be swallowed: workers appended under a mutex, and
    /// the worker's Result was discarded.
    #[test]
    fn chunks_stay_in_file_order_and_errors_are_returned() {
        let first = write_one_chunk(&bson::doc! { "n": 10i32 }, 1, 0, &[]);
        let second = write_one_chunk(&bson::doc! { "n": 20i32 }, 1, 0, &[]);
        let mut bytes = std::fs::read(&first).unwrap();
        bytes.extend(std::fs::read(&second).unwrap());
        let path = std::env::temp_dir().join(format!("ftdc-rs-order-{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(
            series.get("n").map(Vec::as_slice),
            Some([10, 20].as_slice()),
            "chunk 0's sample has to come before chunk 1's"
        );
        let _ = std::fs::remove_file(&first);
        let _ = std::fs::remove_file(&second);
        let _ = std::fs::remove_file(&path);

        let oid = bson::oid::ObjectId::from_bytes([0x22; 12]);
        let reference = bson::doc! { "id": oid, "n": 1i32 };
        let path = write_one_chunk(&reference, 2, 0, &[]);
        let mut parsed = FTDCFileDecoder::new();
        let err = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .expect_err("metric_count mismatch must fail read_and_decode");
        assert!(err.to_string().contains("2"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_to_dataframe() {
        // Create a test HashMap with some data
        let mut test_metrics: HashMap<String, Vec<u64>> = HashMap::new();
        test_metrics.insert("metric1".to_string(), vec![1, 2, 3, 4, 5]);
        test_metrics.insert("metric2".to_string(), vec![10, 20, 30, 40, 50]);
        test_metrics.insert("metric3".to_string(), vec![100, 200, 300]); // Shorter vector to test padding

        // Convert the HashMap to a DataFrame
        let df_result = FTDCFileDecoder::metrics_to_dataframe(test_metrics);

        // Verify the result is Ok
        assert!(df_result.is_ok());

        let df = df_result.unwrap();

        // Verify the DataFrame has the expected structure
        assert_eq!(df.width(), 3); // 3 columns
        assert_eq!(df.height(), 5); // 5 rows (max length of the vectors)

        // Verify column names
        let column_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(column_names.contains(&"metric1".to_string()));
        assert!(column_names.contains(&"metric2".to_string()));
        assert!(column_names.contains(&"metric3".to_string()));

        // Verify the data in the columns
        let metric1_col = df.column("metric1").unwrap();
        let metric2_col = df.column("metric2").unwrap();
        let metric3_col = df.column("metric3").unwrap();

        // Check first values
        assert_eq!(metric1_col.u64().unwrap().get(0).unwrap(), 1);
        assert_eq!(metric2_col.u64().unwrap().get(0).unwrap(), 10);
        assert_eq!(metric3_col.u64().unwrap().get(0).unwrap(), 100);

        // Check that metric3 was padded with zeros
        assert_eq!(metric3_col.u64().unwrap().get(3).unwrap(), 0);
        assert_eq!(metric3_col.u64().unwrap().get(4).unwrap(), 0);
    }
}
