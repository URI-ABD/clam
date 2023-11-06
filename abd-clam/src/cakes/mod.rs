//! CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search.

pub mod knn;
pub mod rnn;
mod search;
mod sharded;
mod singular;

pub use search::Search;
pub use sharded::RandomlySharded;
pub use singular::SingleShard;
