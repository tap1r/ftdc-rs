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
//! The metadata sidecar still stores one string per group: the last type 0
//! document and the last type 2 document, rendered with `Document`'s
//! `Display`. The reader keeps every document. Writing each of them as
//! extended JSON belongs with the command-line export.

use crate::errors::FileDecoderError;
use anyhow::anyhow;
use bson::Document;
use indexmap::IndexMap;
use polars::prelude::*;

/// Build a DataFrame whose columns are the decoded series.
///
/// A missing sample is `None`. Callers align the series first
/// ([`crate::decode::append_chunk`] does this while merging chunks).
pub(crate) fn metrics_to_dataframe(
    metrics: IndexMap<String, Vec<Option<u64>>>,
) -> Result<DataFrame, FileDecoderError> {
    let mut series_vec: Vec<Series> = Vec::new();

    let height = metrics.values().map(Vec::len).max().unwrap_or(0);
    for (key, values) in &metrics {
        if values.len() != height {
            return Err(FileDecoderError::Other(anyhow!(
                "series '{key}' has {} samples but another series has {height}. \
                 A sample that did not exist is null, not 0, and it has to be \
                 inserted on the side where the metric was absent.",
                values.len()
            )));
        }
    }

    for (key, values) in metrics {
        let series = Series::new(key.into(), values);
        series_vec.push(series);
    }

    // DataFrame::from_iter returns a DataFrame directly, not a Result.
    Ok(DataFrame::from_iter(series_vec))
}

/// Write `metrics` to `output_file_name` and, when asked, the JSON sidecar.
pub(crate) fn write_metrics_parquet(
    metrics: IndexMap<String, Vec<Option<u64>>>,
    output_file_name: &str,
    metadata_path: Option<&str>,
    metadata_docs: &[Document],
    metadata_deltas: &[Document],
    document_count: usize,
) -> Result<(), FileDecoderError> {
    let mut df = metrics_to_dataframe(metrics)?;

    let mut file = std::fs::File::create(output_file_name).map_err(|error| {
        FileDecoderError::Other(anyhow!("Failed to create output file: {}", error))
    })?;

    ParquetWriter::new(&mut file)
        .finish(&mut df)
        .map_err(|error| {
            FileDecoderError::Other(anyhow!("Failed to write parquet file: {}", error))
        })?;

    if let Some(metadata_file) = metadata_path {
        let metric_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();

        // The sidecar still stores one string per group. The reader
        // keeps every document; the export that writes them as a JSON
        // array lands with the CLI changes. `Document`'s `Display` is
        // what the previous field stored, so the same text is kept here.
        let metadata = metadata_docs
            .last()
            .map(|doc| doc.to_string())
            .unwrap_or_default();
        let parameters = metadata_deltas
            .last()
            .map(|doc| doc.to_string())
            .unwrap_or_default();
        let json_data = serde_json::json!({
            "metadata": metadata,
            "parameters": parameters,
            "chunks_in_file": document_count,
            "metrics": metric_names,
        });

        let json_string = serde_json::to_string_pretty(&json_data).map_err(|error| {
            FileDecoderError::Other(anyhow!("Failed to serialize metadata to JSON: {}", error))
        })?;

        std::fs::write(metadata_file, json_string).map_err(|error| {
            FileDecoderError::Other(anyhow!("Failed to write metadata file: {}", error))
        })?;
    }

    Ok(())
}
