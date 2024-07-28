//! Extension traits of `Dataset` for specific search applications.

mod searchable;
mod shardable;

pub use searchable::{ParSearchable, Searchable};
pub use shardable::Shardable;
