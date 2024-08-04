//! Extension traits of `Dataset` for specific search applications.

mod compression;
mod searchable;
mod shardable;

use distances::Number;

pub use compression::{Compressible, Encodable};
pub use searchable::{ParSearchable, Searchable};
pub use shardable::Shardable;

use crate::{cluster::ParCluster, Cluster, FlatVec};

impl<I, U: Number, C: Cluster<U>, M> Searchable<I, U, C> for FlatVec<I, U, M> {}

impl<I: Send + Sync, U: Number, C: ParCluster<U>, M: Send + Sync> ParSearchable<I, U, C> for FlatVec<I, U, M> {}
