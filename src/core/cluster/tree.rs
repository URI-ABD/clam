use crate::dataset::Dataset;
use crate::number::Number;

use super::{Cluster, PartitionCriteria};

/// A `Tree` represents a hierarchy of "similar" instances from a
/// metric-`Space`.
///
/// Typically one will chain calls to `new`, `build`, and finally
/// `partition` to construct a fully realized `Tree`.
#[derive(Debug)]
pub struct Tree<T: Number, U: Number, D: Dataset<T, U>> {
    data: D,
    root: Cluster<T, U, D>,
    _t: std::marker::PhantomData<T>,
}

impl<T: Number, U: Number, D: Dataset<T, U>> Tree<T, U, D> {
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
    pub(crate) fn root(&self) -> &Cluster<T, U, D> {
        &self.root
    }

    /// Returns a reference to dataset associated with the tree
    pub fn data(&self) -> &D {
        &self.data
    }

    /// # Returns
    /// The cardinality of the `Tree`
    pub fn cardinality(&self) -> usize {
        self.root.cardinality()
    }

    /// # Returns
    /// The radius of the `Tree`
    pub fn radius(&self) -> U {
        self.root.radius()
    }

    /// # Arguments
    /// criteria: A `PartitionCriteria` through which the `Tree`'s root will be partitioned.
    ///
    /// # Returns
    /// A new `Tree` with a partitioned root.
    pub fn par_partition(mut self, criteria: &PartitionCriteria<T, U, D>, recursive: bool) -> Self {
        self.root = self.root.par_partition(&self.data, criteria, recursive);
        self
    }

    /// Partitions the `Tree` based off of a given criteria
    ///
    /// # Arguments
    /// criteria: A `PartitionCriteria` through which the `Tree`'s root will be partitioned.
    ///
    /// # Returns
    /// A new `Tree` with a partitioned root.
    pub fn partition(mut self, criteria: &PartitionCriteria<T, U, D>, recursive: bool) -> Self {
        self.root = self.root.partition(&self.data, criteria, recursive);
        self
    }

    /// Returns the indices contained in the root of the `Tree`.
    pub fn indices(&self) -> &[usize] {
        self.root.indices(&self.data)
    }

    /// Reorders the `Tree`'s underlying dataset based off of a depth first traversal of a
    /// tree and reformats the tree to reflect the reordering.
    pub fn depth_first_reorder(mut self) -> Self {
        let leaf_indices = self.root.leaf_indices();
        self.data.reorder(&leaf_indices);
        self.root.dfr(&self.data, 0);
        self
    }
}
