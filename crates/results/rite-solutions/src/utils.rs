//! Some utilities

use ftlog::{
    appender::{FileAppender, Period},
    LevelFilter, LoggerGuard,
};

#[allow(dead_code)]
pub fn configure_logger<P: AsRef<std::path::Path>>(log_path: P) -> Result<LoggerGuard, String> {
    let writer = FileAppender::builder().path(&log_path).rotate(Period::Day).build();

    let log_path = log_path.as_ref().to_path_buf();
    let err_path = log_path.with_extension("err.log");

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

    Ok(guard)
}
