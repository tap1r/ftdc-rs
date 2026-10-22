//! MongoDB FTDC diagnostic archives, decoded into columnar metrics.
//!
//! An archive (`metrics.<utc>-<uniquifier>`, or `metrics.interim`) is a
//! concatenation of BSON documents. The modules follow that layout:
//!
//! * `archive` reads each length-prefixed document and records it by `type`.
//!   Type 0 is metadata, type 1 is a compressed metric chunk, type 2 is a
//!   periodic metadata delta. A type this build does not know is skipped.
//! * `decode` inflates a metric chunk and rebuilds each series from the
//!   reference sample plus the delta block.
//! * `extract` walks the reference document and keeps the numeric fields
//!   the server compressor keeps.
//! * `varint` reads one unsigned LEB128 integer from the delta block.
//! * `export` writes the aligned series to Parquet. A sample that did not
//!   exist is null.
//!
//! Chunks are inflated and decoded in parallel. They are appended in file
//! order, which is the order of the samples.

mod archive;
mod decode;
mod errors;
mod export;
mod extract;
mod varint;

pub use archive::FTDCFileDecoder;

#[cfg(test)]
pub(crate) use extract::decimal_text_to_i64;

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
            match crate::extract::extract_metric_meta_data(&chunk) {
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
            match crate::extract::extract_metric_meta_data(&chunk) {
                Ok(mut metrics) => {
                    match crate::decode::extract_all_metrics_from_buffer(&mut metrics, chunk) {
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
            Some([Some(1), Some(2), Some(3), Some(4)].as_slice())
        );
        assert_eq!(
            series.get("x").map(Vec::as_slice),
            Some([Some(2), Some(2), Some(2), Some(2)].as_slice())
        );
        let _ = std::fs::remove_file(&path);

        let reference_only = write_one_chunk(&reference, 2, 0, &[]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(reference_only.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(
            series.get("a").map(Vec::as_slice),
            Some([Some(1)].as_slice())
        );
        assert_eq!(
            series.get("x").map(Vec::as_slice),
            Some([Some(2)].as_slice())
        );
        let _ = std::fs::remove_file(&reference_only);

        let wide = bson::doc! { "n": 0i64 };
        let path = write_one_chunk(&wide, 1, 300, &vec![1u64; 300]);
        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        let values = series.get("n").expect("metric n");
        assert_eq!(values.len(), 301);
        assert_eq!(values[0], Some(0));
        assert_eq!(values[300], Some(300));
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
        assert_eq!(
            series.get("n").map(Vec::as_slice),
            Some([Some(7)].as_slice())
        );
        assert_eq!(
            series.get("ok").map(Vec::as_slice),
            Some([Some(1)].as_slice())
        );
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
        let mut metrics = crate::extract::extract_metric_meta_data(&inflated[0]).unwrap();
        let err = crate::decode::extract_all_metrics_from_buffer(&mut metrics, &inflated[0])
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
            Some([Some(42)].as_slice())
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
            Some([Some(1)].as_slice())
        );
        assert_eq!(
            series.get("members.1.health").map(Vec::as_slice),
            Some([Some(2)].as_slice())
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
            Some([Some(10), Some(20)].as_slice()),
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

    /// A metric that appears in a later chunk is null until then. A metric
    /// that disappears is null afterwards. Neither side is filled with 0.
    #[test]
    fn schema_changes_are_null_not_zero() {
        let only_a = write_one_chunk(&bson::doc! { "a": 1i32 }, 1, 0, &[]);
        let a_and_b = write_one_chunk(&bson::doc! { "a": 2i32, "b": 3i32 }, 2, 0, &[]);
        let only_b = write_one_chunk(&bson::doc! { "b": 4i32 }, 1, 0, &[]);
        let mut bytes = std::fs::read(&only_a).unwrap();
        bytes.extend(std::fs::read(&a_and_b).unwrap());
        bytes.extend(std::fs::read(&only_b).unwrap());
        let path = std::env::temp_dir().join(format!("ftdc-rs-nulls-{}.bin", std::process::id()));
        std::fs::write(&path, bytes).unwrap();

        let mut parsed = FTDCFileDecoder::new();
        let series = parsed
            .read_and_decode(path.to_str().unwrap(), None)
            .unwrap();
        assert_eq!(
            series.get("a").map(Vec::as_slice),
            Some([Some(1), Some(2), None].as_slice())
        );
        assert_eq!(
            series.get("b").map(Vec::as_slice),
            Some([None, Some(3), Some(4)].as_slice())
        );

        let df = FTDCFileDecoder::metrics_to_dataframe(series).unwrap();
        assert_eq!(df.height(), 3);
        assert_eq!(df.column("b").unwrap().u64().unwrap().get(0), None);
        assert_eq!(df.column("a").unwrap().u64().unwrap().get(2), None);

        let _ = std::fs::remove_file(&only_a);
        let _ = std::fs::remove_file(&a_and_b);
        let _ = std::fs::remove_file(&only_b);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_to_dataframe() {
        let mut test_metrics: HashMap<String, Vec<Option<u64>>> = HashMap::new();
        test_metrics.insert(
            "metric1".to_string(),
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        );
        test_metrics.insert(
            "metric2".to_string(),
            vec![Some(10), Some(20), Some(30), Some(40), Some(50)],
        );
        // Already aligned. The nulls are the samples that did not exist.
        // The builder must not turn them into 0.
        test_metrics.insert(
            "metric3".to_string(),
            vec![Some(100), Some(200), Some(300), None, None],
        );

        let df = FTDCFileDecoder::metrics_to_dataframe(test_metrics).unwrap();
        assert_eq!(df.width(), 3);
        assert_eq!(df.height(), 5);

        let metric1_col = df.column("metric1").unwrap();
        let metric2_col = df.column("metric2").unwrap();
        let metric3_col = df.column("metric3").unwrap();
        assert_eq!(metric1_col.u64().unwrap().get(0), Some(1));
        assert_eq!(metric2_col.u64().unwrap().get(0), Some(10));
        assert_eq!(metric3_col.u64().unwrap().get(0), Some(100));
        assert_eq!(metric3_col.u64().unwrap().get(3), None);
        assert_eq!(metric3_col.u64().unwrap().get(4), None);

        let mut ragged = HashMap::new();
        ragged.insert("a".to_string(), vec![Some(1u64), Some(2)]);
        ragged.insert("b".to_string(), vec![Some(3u64)]);
        assert!(FTDCFileDecoder::metrics_to_dataframe(ragged).is_err());
    }

    /// Each document in an archive starts with its BSON length. Ending on a
    /// document boundary is a finished file. Ending inside the 4-byte length,
    /// or inside the body the length promised, is a truncated archive.
    #[test]
    fn truncated_document_is_an_error_and_empty_file_is_empty() {
        let dir = std::env::temp_dir();
        let empty = dir.join(format!("ftdc-rs-empty-{}.bin", std::process::id()));
        std::fs::write(&empty, []).unwrap();
        let mut parsed = FTDCFileDecoder::new();
        parsed.read_file(empty.to_str().unwrap()).unwrap();
        assert_eq!(parsed.document_count, 0);
        let _ = std::fs::remove_file(&empty);

        let short = dir.join(format!("ftdc-rs-short-{}.bin", std::process::id()));
        std::fs::write(&short, [1u8, 0]).unwrap();
        let mut parsed = FTDCFileDecoder::new();
        let err = parsed.read_file(short.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("middle"), "{err}");
        let _ = std::fs::remove_file(&short);

        // A complete metadata document, then a length that claims 50 bytes
        // of which only the length and three more bytes are present.
        let mut bytes = Vec::new();
        bson::doc! { "type": 0i32, "doc": { "host": "a" } }
            .to_writer(&mut bytes)
            .unwrap();
        bytes.extend_from_slice(&50u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 3]);
        let cut = dir.join(format!("ftdc-rs-cut-{}.bin", std::process::id()));
        std::fs::write(&cut, &bytes).unwrap();
        let mut parsed = FTDCFileDecoder::new();
        let err = parsed.read_file(cut.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("middle"), "{err}");
        let _ = std::fs::remove_file(&cut);
    }
}
