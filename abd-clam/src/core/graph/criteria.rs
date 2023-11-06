//! Criteria used for selecting `Cluster`s for `Graph`s.

/// A `Box`ed function that assigns a score for a given `Cluster`.
pub type MetaMLScorer = Box<fn(crate::core::cluster::Ratios) -> f64>;
