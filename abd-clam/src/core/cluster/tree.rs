//! A `Tree` represents a hierarchy of "similar" instances from a metric-`Space`.

use distances::Number;

use crate::{Cluster, Dataset, Instance, PartitionCriteria};

/// A `Tree` represents a hierarchy of `Cluster`s, i.e. "similar" instances
/// from a metric-`Space`.
///
/// # Type Parameters
///
/// - `T`: The type of the instances in the `Tree`.
/// - `U`: The type of the distance values between instances.
/// - `D`: The type of the `Dataset` from which the `Tree` is built.
#[derive(Debug)]
pub struct Tree<I: Instance, U: Number, D: Dataset<I, U>> {
    /// The dataset from which the tree is built.
    pub(crate) data: D,
    /// The root `Cluster` of the tree.
    pub(crate) root: Cluster<U>,
    /// The depth of the tree.
    pub(crate) depth: usize,
    /// To satisfy the `Instance` trait bound.
    _i: core::marker::PhantomData<I>,
}

impl<I: Instance, U: Number, D: Dataset<I, U>> Tree<I, U, D> {
    /// Constructs a new `Tree` for a given dataset. Importantly, this does not
    /// partition the tree.
    ///
    /// # Arguments
    /// dataset: The dataset from which the tree will be built
    pub fn new(data: D, seed: Option<u64>) -> Self {
        let root = Cluster::new_root(&data, seed);
        let depth = root.max_leaf_depth();
        Self {
            data,
            root,
            depth,
            _i: core::marker::PhantomData,
        }
    }

    /// Recursively partitions the root `Cluster` using the given criteria.
    ///
    /// # Arguments
    ///
    /// * `criteria`: the criteria used to decide when to partition a `Cluster`.
    ///
    /// # Returns
    ///
    /// The `Tree` after partitioning.
    #[must_use]
    pub fn partition(mut self, criteria: &PartitionCriteria<U>) -> Self {
        self.root = self.root.partition(&mut self.data, criteria);
        self
    }

    /// Returns a reference to the data used to build the `Tree`.
    pub const fn data(&self) -> &D {
        &self.data
    }

    /// The cardinality of the `Tree`, i.e. the number of instances in the data.
    pub const fn cardinality(&self) -> usize {
        self.root.cardinality
    }

    /// The radius of the root of the `Tree`.
    pub const fn radius(&self) -> U {
        self.root.radius
    }

    // /// Returns the indices contained in the root of the `Tree`.
    // pub fn indices(&self) -> &[usize] {
    //     self.data.indices()
    // }
}
