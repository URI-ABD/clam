use distances::Number;

use crate::dataset::Dataset;

use super::{Cluster, PartitionCriteria};

/// A `Tree` represents a hierarchy of "similar" instances from a
/// metric-`Space`.
///
/// Typically one will chain calls to `new`, `build`, and finally
/// `partition` to construct a fully realized `Tree`.
#[derive(Debug)]
pub struct Tree<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> {
    data: D,
    root: Cluster<T, U>,
    _t: std::marker::PhantomData<T>,
}

impl<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> Tree<T, U, D> {
    /// Constructs a new `Tree` for a given dataset. Importantly,
    /// this does not build nor partition the tree.
    ///
    /// # Arguments
    /// dataset: The dataset from which the tree will be built
    pub fn new(data: D, seed: Option<u64>) -> Self {
        Tree {
            root: Cluster::new_root(&data, data.indices(), seed),
            data,
            _t: Default::default(),
        }
    }

    /// # Returns
    /// A reference to the root `Cluster` of the tree
    pub(crate) fn root(&self) -> &Cluster<T, U> {
        &self.root
    }

    /// Returns a reference to dataset associated with the tree
    pub fn data(&self) -> &D {
        &self.data
    }

    /// # Returns
    /// The cardinality of the `Tree`
    pub fn cardinality(&self) -> usize {
        self.root.cardinality
    }

    /// # Returns
    /// The radius of the `Tree`
    pub fn radius(&self) -> U {
        self.root.radius
    }

    /// Partitions the `Tree` based off of a given criteria
    ///
    /// # Arguments
    /// criteria: A `PartitionCriteria` through which the `Tree`'s root will be partitioned.
    ///
    /// # Returns
    /// A new `Tree` with a partitioned root.
    pub fn partition(mut self, criteria: &PartitionCriteria<T, U>) -> Self {
        self.root = self.root.partition(&mut self.data, criteria);
        self
    }

    /// Returns the indices contained in the root of the `Tree`.
    pub fn indices(&self) -> &[usize] {
        self.data.indices()
    }
}
