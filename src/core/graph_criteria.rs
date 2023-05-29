//! Criteria used for selecting `Cluster`s for `Graph`s.

/// A `Box`ed function that assigns a score for a given `Cluster`.
pub type MetaMLScorer = Box<dyn (Fn(super::cluster::Ratios) -> f64) + Send + Sync>;
