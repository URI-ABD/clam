//! The datasets we use for benchmarks.

mod raw;
mod tree;

#[allow(clippy::module_name_repetitions)]
#[allow(unused_imports)]
pub use raw::{fasta, RawData};

#[allow(unused_imports)]
pub use tree::PathManager;
