//! Parquet export of decoded FTDC series.
//!
//! Each metric is a column of `u64` values. A missing sample is `None`,
//! which Parquet stores as a null. Many FTDC counters sit at 0 for long
//! stretches, so a filler zero cannot be told apart from a real sample.
//! Columns are written in the map's iteration order. That order is the
//! order names were first seen, which is the reference document's field
//! order.
//! Columns of one file share one height. A map whose columns differ in
//! length is rejected. The merge in `decode` is what inserts the nulls
//! on the side where the metric was absent.
//!
//! The sidecar is relaxed extended JSON. `metadata` is every type 0
//! wrapper (`_id` and `doc`). `metadataDeltas` is every type 2 wrapper
//! (`_id`, `count`, and `doc`), in file order. Dates, ObjectIds, and
//! timestamps keep their extended-JSON forms (`$date`, `$oid`,
//! `$timestamp`). Integers and strings stay JSON values. `count` is the
//! delta encoder's generation, not a sample index. An earlier sidecar kept
//! only the inner `doc`, and before that only the last document of each
//! kind, rendered with `Document`'s `Display`. `metricTypes` names the
//! compressor rule for each column. The Parquet values stay `u64`.

use crate::errors::FileDecoderError;
use crate::extract::MetricColumn;
use indexmap::IndexMap;
use polars::prelude::*;
use std::path::Path;

/// Build a DataFrame whose columns are the decoded series.
///
/// A missing sample is `None`. Callers align the series first
/// ([`crate::decode::append_chunk`] does this while merging chunks).
pub(crate) fn metrics_to_dataframe(
    metrics: IndexMap<String, MetricColumn>,
) -> Result<DataFrame, FileDecoderError> {
    let mut series_vec: Vec<Series> = Vec::new();

    let height = metrics
        .values()
        .map(|column| column.values.len())
        .max()
        .unwrap_or(0);
    for (key, column) in &metrics {
        if column.values.len() != height {
            return Err(FileDecoderError::format(format!(
                "series '{key}' has {} samples but another series has {height}. \
                 A sample that did not exist is null, not 0, and it has to be \
                 inserted on the side where the metric was absent.",
                column.values.len()
            )));
        }
    }

    for (key, column) in metrics {
        let series = Series::new(key.into(), column.values);
        series_vec.push(series);
    }

    // DataFrame::from_iter returns a DataFrame directly, not a Result.
    Ok(DataFrame::from_iter(series_vec))
}

/// Write `metrics` to `output_file_name` and, when asked, the JSON sidecar.
pub(crate) fn write_metrics_parquet(
    metrics: IndexMap<String, MetricColumn>,
    output_file_name: &Path,
    metadata_path: Option<&Path>,
    mut preamble: serde_json::Value,
) -> Result<(), FileDecoderError> {
    // Names and kinds come from the map before it is moved into the frame.
    // IndexMap iteration is first-seen order, which is also the order
    // `metrics_to_dataframe` writes the columns. The kind is the compressor
    // rule (`int`, `double`, `decimal`, `bool`, `date`, `timestampSeconds`,
    // `timestampIncrement`). The cell values stay the `u64` bit patterns.
    let metric_names: Vec<String> = metrics.keys().cloned().collect();
    let metric_types: serde_json::Map<String, serde_json::Value> = metrics
        .iter()
        .map(|(name, column)| {
            (
                name.clone(),
                serde_json::Value::String(column.kind.as_str().to_string()),
            )
        })
        .collect();

    let mut df = metrics_to_dataframe(metrics)?;

    let mut file = std::fs::File::create(output_file_name)
        .map_err(|error| FileDecoderError::io("failed to create", output_file_name, error))?;

    ParquetWriter::new(&mut file)
        .finish(&mut df)
        .map_err(|error| {
            FileDecoderError::format(format!(
                "failed to write Parquet {}: {error}",
                output_file_name.display()
            ))
        })?;

    if let Some(metadata_file) = metadata_path {
        // Column order matches the Parquet file. `documentCount` counts
        // every BSON document in the archive, including metric chunks and
        // types this build skipped. `skippedChunkTypes` is that list.
        // The preamble already holds the type 0 and type 2 wrappers.
        let Some(object) = preamble.as_object_mut() else {
            return Err(FileDecoderError::format(format!(
                "metadata sidecar preamble is not a JSON object"
            )));
        };
        object.insert("metrics".to_string(), serde_json::json!(metric_names));
        object.insert(
            "metricTypes".to_string(),
            serde_json::Value::Object(metric_types),
        );

        let json_string = serde_json::to_string_pretty(&preamble).map_err(|error| {
            FileDecoderError::format(format!("failed to serialize metadata to JSON: {error}"))
        })?;

        std::fs::write(metadata_file, json_string)
            .map_err(|error| FileDecoderError::io("failed to write", metadata_file, error))?;
    }

    Ok(())
}
