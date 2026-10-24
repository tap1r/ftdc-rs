//! Which fields of a reference document become FTDC metrics.
//!
//! The compressor walks the document in field order and appends a `u64`
//! for each value whose BSON type it knows how to store:
//!
//! * `double`, `int32`, `int64`, `decimal128`, `bool`, `datetime` — one
//!   `u64` each
//! * `timestamp` — two `u64`s, seconds then increment
//! * subdocuments — walked, in the order of their fields
//! * arrays — walked as documents whose keys are `"0"`, `"1"`, ... so an
//!   element's path is `parent.key.<index>.field`
//!
//! Strings, ObjectIds, binary, null, regex, JavaScript, symbol, undefined,
//! MinKey, MaxKey, and DBPointer are skipped. Two ObjectIds show up in real
//! `serverStatus` output and are the usual reason a hand-rolled walk is
//! longer than the chunk header: `repl.topologyVersion.processId` on every
//! sample, and `repl.electionId` after an election. The header's
//! `metric_count` is the length of the server's walk. Deltas are positional,
//! so a longer walk applies every later delta to the wrong series.
//!
//! Decimal128 has no integer accessor in the `bson` crate (it exposes
//! `Display`: `"42.9"`, `"1.05E+3"`, `"NaN"`, `"Infinity"`). The compressor
//! calls `BSONElement::numberLong()`, which truncates toward zero and
//! saturates at the int64 limits. The integer is parsed from that text, and
//! the int64 bit pattern is the metric. A negative value is therefore a
//! large unsigned number, which is what a later wrapping delta expects.
//!
//! A double is squeezed into an int64 before that same bit cast. NaN
//! becomes 0. A value greater than or equal to 2^63, including +Infinity,
//! becomes `i64::MAX`. A value less than -2^63, including -Infinity,
//! becomes `i64::MIN`. Every other value is truncated toward zero. See
//! [`double_to_metric`].
//!
//! The path of a field is its own name at the root, and `parent.key` under
//! a document or an array element. Array elements use the decimal index as
//! the key (`members.0.health`). A timestamp's two metrics are `field_time`
//! and `field_increment`, seconds then increment.
//!
//! Each metric also keeps the rule that produced it ([`MetricKind`]). The
//! stored sample is still the `u64` bit pattern. The kind is how a caller
//! tells a counter, a truncated double, a boolean, and a timestamp apart.

use crate::errors::FileDecoderError;
use bson::{Bson, Document};

/// Which compressor rule produced a metric's `u64` bits.
///
/// int32 and int64 share [`MetricKind::Int`]: both are `numberLong`'s bit
/// pattern. Decimal128 is separate. It also ends as `numberLong`, but the
/// conversion is the decimal truncation, not the double arm and not an
/// integer load. A timestamp is two columns, seconds then increment, and
/// the two kinds stay in that order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    Double,
    Int,
    Decimal,
    Bool,
    Date,
    TimestampSeconds,
    TimestampIncrement,
}

impl MetricKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Int => "int",
            Self::Decimal => "decimal",
            Self::Bool => "bool",
            Self::Date => "date",
            Self::TimestampSeconds => "timestampSeconds",
            Self::TimestampIncrement => "timestampIncrement",
        }
    }
}

/// One decoded metric: the compressor's kind, and one sample per row.
///
/// `values` are the `u64` bit patterns. A missing sample is `None`. `kind`
/// is taken from the chunk that introduced the name. A later chunk that
/// classifies the same name with a different kind is an error: the sidecar
/// would otherwise describe the column as something the later samples are not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricColumn {
    pub kind: MetricKind,
    pub values: Vec<Option<u64>>,
}

#[derive(Debug)]
pub(crate) struct MetricData {
    pub(crate) name: String,
    pub(crate) initial: u64,
    pub(crate) kind: MetricKind,
    /// BSON value this sample was taken from. Kept so a metric_count
    /// mismatch can say which field disagreed with the compressor.
    #[allow(dead_code)]
    pub(crate) metric_type: Bson,
    pub(crate) metrics: Option<usize>,
    /// Reconstructed samples for one chunk.
    ///
    /// Index 0 is the value taken from the reference document. Each later
    /// index is that value plus the deltas that follow it. The length is
    /// `sample_count + 1`, not `sample_count`: the header counts deltas,
    /// and the reference document is itself a sample.
    pub(crate) data: Option<Vec<u64>>,
}

pub(crate) fn push_metric(
    name: &str,
    initial: u64,
    kind: MetricKind,
    metric_type: Bson,
    out: &mut Vec<MetricData>,
) {
    out.push(MetricData {
        name: name.to_string(),
        initial,
        kind,
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
/// is then stored as the uint64 metric.
pub(crate) fn decimal128_to_metric(value: &bson::Decimal128) -> u64 {
    decimal_text_to_i64(&value.to_string()) as u64
}

pub(crate) fn decimal_text_to_i64(text: &str) -> i64 {
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

/// Join `key` onto `parent` the way the reference walk names a metric.
///
/// An empty parent means this field is at the root, so the name is `key`
/// itself. A leading dot used to be prefixed here and stripped again when
/// the column was published. `name.remove(0)` also deletes the first
/// character of a name that did not have that dot.
fn metric_path(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    }
}

/// Reduce a double the way `FTDCBSONUtil::extractMetricsFromDocument` does.
///
/// The server's comment on this conversion is the rule. Doubles that fall
/// outside int64 become:
///
/// * NaN → 0
/// * +Infinity, and any value `>= 2^63`, → `i64::MAX`
/// * -Infinity, and any value `< -2^63`, → `i64::MIN`
///
/// `2^63` is `BSONElement::kLongLongMaxPlusOneAsDouble`. The comparison in
/// the server is `!(value < kLongLongMaxPlusOneAsDouble)`, after NaN has
/// already been handled. `i64::MAX as f64` is that same `2^63`, because
/// `2^63 - 1` is not representable and rounds up.
///
/// Every remaining value is truncated toward zero (`static_cast<long long>`).
/// The int64 bit pattern is what gets stored in the `uint64_t` metric vector.
/// Casting the double straight to `u64` turns a negative into 0 and a value
/// past `2^63` into `u64::MAX` or the raw magnitude. Deltas were computed
/// from the int64 bits, so that reference is the wrong base.
pub(crate) fn double_to_metric(value: f64) -> u64 {
    if value.is_nan() {
        return 0;
    }
    let reduced = if !(value < i64::MAX as f64) {
        i64::MAX
    } else if value < i64::MIN as f64 {
        i64::MIN
    } else {
        value as i64
    };
    reduced as u64
}

/// Walk `root` in field order and append one [`MetricData`] per metric.
pub(crate) fn extract_metric_data_from_doc(
    doc_name: &str,
    root: Document,
    metric_data: &mut Vec<MetricData>,
) {
    for (key, value) in root {
        let path = metric_path(doc_name, &key);
        match value {
            Bson::Double(v) => {
                push_metric(
                    &path,
                    double_to_metric(v),
                    MetricKind::Double,
                    value,
                    metric_data,
                );
            }
            Bson::Int32(v) => push_metric(&path, v as u64, MetricKind::Int, value, metric_data),
            Bson::Int64(v) => push_metric(&path, v as u64, MetricKind::Int, value, metric_data),
            Bson::Decimal128(v) => {
                // `BSONElement::numberLong()` for NumberDecimal, which is
                // what the compressor calls. One uint64, not a sentinel.
                // The sentinel 999999999 used to be the base every delta
                // in this column was added to.
                push_metric(
                    &path,
                    decimal128_to_metric(&v),
                    MetricKind::Decimal,
                    value,
                    metric_data,
                );
            }
            Bson::Boolean(v) => {
                push_metric(&path, u64::from(v), MetricKind::Bool, value, metric_data)
            }
            Bson::DateTime(v) => {
                push_metric(
                    &path,
                    v.timestamp_millis() as u64,
                    MetricKind::Date,
                    value,
                    metric_data,
                );
            }
            Bson::Timestamp(v) => {
                // Two metrics, seconds then increment. The names gain a
                // suffix so the two columns can be told apart; the order
                // is the order the compressor pushes them.
                push_metric(
                    &format!("{path}_time"),
                    u64::from(v.time),
                    MetricKind::TimestampSeconds,
                    value.clone(),
                    metric_data,
                );
                push_metric(
                    &format!("{path}_increment"),
                    u64::from(v.increment),
                    MetricKind::TimestampIncrement,
                    value,
                    metric_data,
                );
            }
            Bson::Document(sub_doc) => {
                extract_metric_data_from_doc(&path, sub_doc, metric_data);
            }
            Bson::Array(items) => {
                for (index, item) in items.into_iter().enumerate() {
                    // Re-enter through the same match by wrapping the
                    // element in a one-field document whose key is the
                    // index. That keeps array elements and document
                    // fields on one path.
                    let mut wrapper = Document::new();
                    wrapper.insert(index.to_string(), item);
                    extract_metric_data_from_doc(&path, wrapper, metric_data);
                }
            }
            // String, ObjectId, Null, Binary, Regex, JavaScript, Symbol,
            // Undefined, MinKey, MaxKey, DBPointer. Not metrics.
            _ => {}
        }
    }
}

/// Parse the reference document at the front of an inflated chunk and walk it.
///
/// The reference document is a BSON document, so its own first four bytes
/// are its length. Bytes after that document are the chunk header and the
/// deltas; `Document::from_reader` stops at the end of the document.
pub(crate) fn extract_metric_meta_data(
    buffer: &Vec<u8>,
) -> Result<Vec<MetricData>, FileDecoderError> {
    let mut base = buffer.as_slice();

    match Document::from_reader(&mut base) {
        Ok(bson_doc) => {
            let mut ret_metrics: Vec<MetricData> = vec![];
            extract_metric_data_from_doc("", bson_doc, &mut ret_metrics);
            Ok(ret_metrics)
        }
        Err(e) => Err(FileDecoderError::format(e.to_string())),
    }
}
