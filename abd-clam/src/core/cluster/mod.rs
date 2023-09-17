//! Provides the `Cluster` struct, which is used to store the cluster data.
//!
//! It also provides the `Tree` struct, which is used to store a tree of clusters
//! and is meant to be public.
//!
//! It also provides the `PartitionCriteria` trait, and implementations for
//! `PartitionCriteria` for `MaxDepth` and `MinCardinality` which are used to
//! determine when to stop partitioning the tree.
mod _cluster;
mod criteria;
mod tree;

#[allow(clippy::module_name_repetitions)]
pub use _cluster::{Cluster, SerializedChildInfo, SerializedCluster};
pub use criteria::{PartitionCriteria, PartitionCriterion};
pub use tree::Tree;
