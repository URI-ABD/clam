//! Some utilities

use std::path::PathBuf;

use ftlog::{
    appender::{FileAppender, Period},
    LevelFilter, LoggerGuard,
};

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
    let log_path = logs_dir.join(format!("{file_name}.log"));

    let writer = FileAppender::builder().path(&log_path).rotate(Period::Day).build();

    let err_path = log_path.with_extension("err.log");

    let guard = ftlog::Builder::new()
        // global max log level
        .max_log_level(LevelFilter::Trace)
        // define root appender, pass None would write to stderr
        .root(writer)
        // write `Warn` and `Error` logs in ftlog::appender to `err_path` instead of `log_path`
        .filter("ftlog::appender", "ftlog-appender", LevelFilter::Warn)
        .appender("ftlog-appender", FileAppender::new(err_path))
        .try_init()
        .map_err(|e| e.to_string())?;

    Ok((guard, log_path))
}
