///TODO! Add documentation on graph module
mod _graph;

/// Helper functions for building graph
/// Selects clusters and detects edges
mod builder;

pub mod criteria;
pub use _graph::{Edge, Graph};

pub use criteria::MetaMLScorer;
