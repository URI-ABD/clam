//! Some utilities

use std::path::PathBuf;

use ftlog::{
    LevelFilter, LoggerGuard,
    appender::{FileAppender, Period},
};

use crate::{data::ShellData, metrics::Metric};

/// Creates a path for the tree file based on the data type, metric type, and optional prefix and suffix.
pub fn tree_file_path<P: AsRef<std::path::Path>>(out_dir: P, data: &ShellData, metric: &Metric, prefix: Option<&str>, suffix: Option<&str>) -> PathBuf {
    let data_part = match data {
        ShellData::String(_) => "str",
        _ => "vec",
    };
    let metric_part = match metric {
        Metric::Lcs => "lcs",
        Metric::Levenshtein => "lev",
        Metric::Euclidean => "euc",
        Metric::Cosine => "cos",
    };

    let prefix = prefix.map_or("".to_string(), |f| format!("{f}-"));
    let middle = format!("tree-{data_part}-{metric_part}");
    let suffix = suffix.map_or("".to_string(), |s| format!("-{s}"));

    out_dir.as_ref().join(format!("{prefix}{middle}{suffix}.bin"))
}

/// Configures the logger.
///
/// # Errors
///
/// - If a logs directory could not be located/created.
/// - If the logger could not be initialized.
pub fn configure_logger(file_name: &str) -> Result<(LoggerGuard, PathBuf), String> {
    let root_dir = PathBuf::from(".").canonicalize().map_err(|e| e.to_string())?;
    let logs_dir = root_dir.join("logs");
    if !logs_dir.exists() {
        std::fs::create_dir(&logs_dir).map_err(|e| e.to_string())?;
    }
    let log_path = logs_dir.join(file_name);

    let writer = FileAppender::builder().path(&log_path).rotate(Period::Day).build();

    let err_stem = log_path.file_stem().unwrap().to_str().unwrap();
    let err_stem = format!("{err_stem}-err");
    let err_path = log_path.with_file_name(err_stem);

    let guard = ftlog::Builder::new()
        // global max log level
        .max_log_level(LevelFilter::Info)
        // define root appender, pass None would write to stderr
        .root(writer)
        // write `Warn` and `Error` logs in ftlog::appender to `err_path` instead of `log_path`
        .filter("ftlog::appender", "ftlog-appender", LevelFilter::Warn)
        .appender("ftlog-appender", FileAppender::new(err_path))
        .try_init()
        .map_err(|e| e.to_string())?;

    Ok((guard, log_path))
}
