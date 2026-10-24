use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Failure while reading an FTDC archive or writing its Parquet output.
///
/// A filesystem problem keeps the path and the OS error. Everything else is
/// a layout problem: a truncated BSON document, a varint that runs off the
/// end of a delta block, a `metric_count` that does not match the reference
/// walk. The message names the rule that failed.
#[derive(Debug, Error)]
pub enum FileDecoderError {
    #[error("{action} {path}: {source}")]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("{0}")]
    Format(String),
}

impl FileDecoderError {
    pub(crate) fn io(action: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            action,
            path: path.into(),
            source,
        }
    }

    pub(crate) fn format(message: impl Into<String>) -> Self {
        Self::Format(message.into())
    }
}
