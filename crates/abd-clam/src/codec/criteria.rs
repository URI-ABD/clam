//! Compression criteria.

use distances::number::UInt;

use crate::Cluster;

use super::SquishyBall;

/// A criterion used to decide whether to compress a `SquishyBall`.
pub trait CompressionCriterion<U: UInt>: Send + Sync {
    /// Check whether a `SquishyBall` meets the criterion for compression.
    fn check(&self, b: &SquishyBall<U>) -> bool;
}

/// Compress all `SquishyBall`s at a fixed depth.
#[derive(Debug, Clone)]
pub struct FixedDepth(usize);

impl<U: UInt> CompressionCriterion<U> for FixedDepth {
    fn check(&self, b: &SquishyBall<U>) -> bool {
        b.depth() == self.0
    }
}

/// Compress all `SquishyBall`s with a cardinality below a threshold.
#[derive(Debug, Clone)]
pub struct MaxCardinality(usize);

impl<U: UInt> CompressionCriterion<U> for MaxCardinality {
    fn check(&self, b: &SquishyBall<U>) -> bool {
        b.cardinality() > self.0
    }
}

/// A collection of criteria used to decide whether to compress a `SquishyBall`.
#[allow(clippy::module_name_repetitions)]
pub struct CompressionCriteria<U: UInt> {
    /// The criteria used to decide whether to compress a `SquishyBall`.
    criteria: Vec<Box<dyn CompressionCriterion<U>>>,
    /// Whether all criteria must be met for a `SquishyBall` to be compressed or if any one criterion
    /// is sufficient.
    check_all: bool,
}

impl<U: UInt> CompressionCriterion<U> for CompressionCriteria<U> {
    fn check(&self, ball: &SquishyBall<U>) -> bool {
        if self.check_all {
            self.criteria.iter().all(|c| c.check(ball))
        } else {
            self.criteria.iter().any(|c| c.check(ball))
        }
    }
}

impl<U: UInt> Default for CompressionCriteria<U> {
    fn default() -> Self {
        Self::new(true).with_fixed_depth(4)
    }
}

impl<U: UInt> CompressionCriteria<U> {
    /// Create a new `CompressionCriteria`.
    pub fn new(check_all: bool) -> Self {
        Self {
            criteria: Vec::new(),
            check_all,
        }
    }

    /// Add a `FixedDepth` criterion to the `CompressionCriteria`.
    #[must_use]
    pub fn with_fixed_depth(mut self, depth: usize) -> Self {
        self.criteria.push(Box::new(FixedDepth(depth)));
        self
    }

    /// Add a `MaxCardinality` criterion to the `CompressionCriteria`.
    #[must_use]
    pub fn with_max_cardinality(mut self, cardinality: usize) -> Self {
        self.criteria.push(Box::new(MaxCardinality(cardinality)));
        self
    }

    /// Add a criterion to the `CompressionCriteria`.
    pub fn with_custom<C: CompressionCriterion<U> + 'static>(mut self, c: C) -> Self {
        self.criteria.push(Box::new(c));
        self
    }
}
