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

pub use _cluster::Cluster;
pub use criteria::PartitionCriteria;
pub use tree::Tree;
