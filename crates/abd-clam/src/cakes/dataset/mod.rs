//! Extension traits of `Dataset` for specific search applications.

mod searchable;
mod shardable;

use distances::Number;
pub use searchable::{ParSearchable, Searchable};
pub use shardable::Shardable;

use crate::{cluster::ParCluster, Cluster, FlatVec};

impl<I, U: Number, C: Cluster<I, U, Self>, M> Searchable<I, U, C> for FlatVec<I, U, M> {}

impl<I: Send + Sync, U: Number, C: ParCluster<I, U, Self>, M: Send + Sync> ParSearchable<I, U, C> for FlatVec<I, U, M> {}
