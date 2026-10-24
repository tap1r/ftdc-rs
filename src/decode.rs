//! One inflated metric chunk, and the series built by appending chunks.
//!
//! A type 1 document stores its payload in `data` as BinData. The first four
//! bytes of that blob are a little-endian `uint32`, the uncompressed size
//! (`metrics_chunk.uncompressed_size`). The rest is a zlib stream. Those four
//! bytes are not zlib bytes.
//!
//! After inflation the layout is:
//!
//! ```text
//! reference document   BSON. Its first 4 bytes are its own length.
//! metric_count         uint32. How many numbers the compressor extracted.
//! sample_count         uint32. How many deltas follow, per metric.
//! deltas               unsigned LEB128, column-major.
//! ```
//!
//! `sample_count` does not include the reference document. Each metric has
//! `sample_count + 1` samples: the reference value, then one reconstructed
//! value per delta. A chunk that only holds the reference (common at the
//! start of a file, and whenever the schema changes before another sample
//! is collected) has `sample_count` 0 and still has one sample.
//!
//! The delta block is column-major. Every delta of metric 0 comes first,
//! then every delta of metric 1. A delta is added to the previous sample
//! with wrapping `u64` addition, which is what the C++ compressor does
//! (`uint64_t`). A delta of `u64::MAX` takes a counter backwards by one.
//! Checked addition would reject those samples.
//!
//! Zeros are run-length encoded across the whole block, including across
//! the boundary between two metrics. A run of *k* zeros is the pair
//! `[0, k - 1]`: the `0` is the first zero, and the next varint is how
//! many further zeros follow. The counter is shared across metrics on
//! purpose. Resetting it at each metric would swallow a run the compressor
//! wrote as one pair.
//!
//! The server refuses a chunk whose `metric_count * sample_count` exceeds
//! 1_000_000, and an inflated payload larger than 10_000_000 bytes. The
//! product is checked here, before the sample vectors are allocated.
//!
//! `metric_count` is authoritative. The reference walk has to produce that
//! many metrics. The check happens before any delta is applied, and the
//! error names the first and last extracted field so a mismatched walk
//! (an ObjectId counted as a metric, an array element skipped) can be
//! seen directly.
//!
//! Chunks of one file inflate and decode independently. Their columns are
//! appended afterwards, in file order. A metric that first appears in a
//! later chunk is null for the rows already emitted. A metric that is
//! absent from a later chunk is null for that chunk's height. Every column
//! of one chunk has the same height, because one chunk is one rectangle of
//! `metric_count` columns by `sample_count + 1` rows. The server closes a
//! chunk when the number of metrics changes, so the next chunk is where a
//! new or dropped series shows up.

use crate::errors::FileDecoderError;
use crate::extract::MetricData;
use crate::varint::{RangeCursor, unpack};
use anyhow::anyhow;
use indexmap::IndexMap;
use std::collections::HashSet;
use zune_inflate::DeflateDecoder;

/// The server refuses a chunk whose `metric_count * sample_count` exceeds this.
const MAX_METRIC_SAMPLES: usize = 1_000_000;

/// Inflate one metric blob.
///
/// The first four bytes are the uncompressed size the server writes in
/// front of the zlib stream. A blob shorter than that is not a metric chunk.
pub(crate) fn inflate_owned(bytes: Vec<u8>) -> Result<Vec<u8>, FileDecoderError> {
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

/// Read the reference length, `metric_count`, and `sample_count`.
///
/// `data_size` is the BSON length of the reference document. The two counts
/// sit in the eight bytes immediately after it. A buffer that cannot hold
/// those eight bytes is truncated.
pub(crate) fn read_header(data: &[u8]) -> Result<(usize, usize, usize), FileDecoderError> {
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
    if metric_count.saturating_mul(sample_count) > MAX_METRIC_SAMPLES {
        return Err(FileDecoderError::Other(anyhow!(
            "metric_count {metric_count} and sample_count {sample_count} exceed the FTDC limit"
        )));
    }
    Ok((data_size, metric_count, sample_count))
}

/// Rebuild every metric in one inflated chunk.
///
/// The returned count is the number of reconstructed sample values
/// (`metrics * (sample_count + 1)`). Each `MetricData::data` vector has
/// length `sample_count + 1`.
pub(crate) fn extract_all_metrics_from_buffer(
    metric_data_vec: &mut Vec<MetricData>,
    buffer: &Vec<u8>,
) -> Result<usize, FileDecoderError> {
    let (doc_size, metric_count, delta_count) = read_header(buffer.as_slice())?;

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
    let mut data_range_cursor = RangeCursor::new(buffer[offset..].to_vec());

    let mut count = 0;
    // Additional zeros still to emit. This survives from one metric
    // to the next. See the module docs for the `[0, k - 1]` pair.
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

/// Reconstruct one inflated chunk as `(name, samples)` pairs, in walk order.
///
/// The name is the path `extract` built. Nothing is stripped. This function
/// does not append to a shared map: the caller appends, in chunk order, and
/// the order of these pairs is the order those names are first recorded.
pub(crate) fn decode_one_chunk(
    chunk: Vec<u8>,
) -> Result<Vec<(String, Vec<u64>)>, FileDecoderError> {
    let mut metric_data = crate::extract::extract_metric_meta_data(&chunk)?;
    extract_all_metrics_from_buffer(&mut metric_data, &chunk)?;

    let mut columns = Vec::with_capacity(metric_data.len());
    for metric in metric_data {
        let Some(samples) = metric.data else {
            return Err(FileDecoderError::Other(anyhow!(
                "metric '{}' produced no samples",
                metric.name
            )));
        };
        columns.push((metric.name, samples));
    }
    Ok(columns)
}

/// Append one chunk's columns to the file-level series.
///
/// `rows` is the number of samples already stored. Every column of this
/// chunk has the same length, because one chunk is one rectangular
/// block: `sample_count + 1` rows and `metric_count` columns.
///
/// A name that was not in earlier chunks gets `rows` leading nulls, then
/// its values. Those nulls are "this metric did not exist yet", which is
/// what the server does when the schema changes and it closes the chunk.
/// Putting the new values at the start of the column, and filling the
/// tail with 0, lines the new metric up with the wrong time and looks
/// like a real zero.
///
/// A name that was in earlier chunks but is absent from this one gets
/// the same number of trailing nulls. The schema changed the other way:
/// the metric went away. A zero there would look like the counter reset.
///
/// Names stay in the order they were first inserted. That is the reference
/// document's field order for the first chunk, then any metric a later
/// chunk introduces, in that chunk's field order. Parquet columns follow
/// this order. A `HashMap` reshuffles them on every run.
pub(crate) fn append_chunk(
    merged: &mut IndexMap<String, Vec<Option<u64>>>,
    rows: &mut usize,
    columns: Vec<(String, Vec<u64>)>,
) -> Result<(), FileDecoderError> {
    let height = columns
        .first()
        .map(|(_, samples)| samples.len())
        .unwrap_or(0);
    let mut present = HashSet::with_capacity(columns.len());

    for (name, samples) in columns {
        if samples.len() != height {
            return Err(FileDecoderError::Other(anyhow!(
                "metric '{name}' has {} samples in a chunk whose other metrics have {height}",
                samples.len()
            )));
        }
        present.insert(name.clone());
        match merged.get_mut(&name) {
            Some(existing) => existing.extend(samples.into_iter().map(Some)),
            None => {
                let mut series = Vec::with_capacity(*rows + height);
                series.resize(*rows, None);
                series.extend(samples.into_iter().map(Some));
                merged.insert(name, series);
            }
        }
    }

    let end = *rows + height;
    for (name, series) in merged.iter_mut() {
        if !present.contains(name) {
            series.resize(end, None);
        }
    }
    *rows = end;
    Ok(())
}
