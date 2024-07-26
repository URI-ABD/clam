//! A `Tree` represents a hierarchy of "similar" instances from a metric-`Space`.

use core::marker::PhantomData;

use std::path::Path;

use distances::Number;

use crate::{Cluster, Dataset, Instance, PartitionCriterion};

/// A `Tree` represents a hierarchy of `Cluster`s, i.e. "similar" instances
/// from a metric-`Space`.
///
/// The `Tree` has other implementation blocks spread across the crate. These
/// are used for specific functionality for the concrete `Cluster` types we provide.
///
/// # Type Parameters
///
/// - `T`: The type of the instances in the `Tree`.
/// - `U`: The type of the distance values between instances.
/// - `D`: The type of the `Dataset` from which the `Tree` is built.
#[derive(Debug)]
pub struct Tree<I: Instance, U: Number, D: Dataset<I, U>, C: Cluster<U>> {
    /// The dataset from which the tree is built.
    pub(crate) data: D,
    /// The root `Cluster` of the tree.
    pub(crate) root: C,
    /// The depth of the tree.
    pub(crate) depth: usize,
    /// To satisfy the `Instance` trait bound.
    _i: PhantomData<I>,
    /// To satisfy the `Number` trait bound.
    _u: PhantomData<U>,
}

impl<I: Instance, U: Number, D: Dataset<I, U>, C: Cluster<U>> Tree<I, U, D, C> {
    /// Constructs a new `Tree` for a given dataset. Importantly, this does not
    /// partition the tree.
    ///
    /// # Arguments
    /// dataset: The dataset from which the tree will be built
    pub fn new(data: D, seed: Option<u64>) -> Self {
        let root = C::new_root(&data, seed);
        let depth = root.max_leaf_depth();
        Self {
            data,
            root,
            depth,
            _i: PhantomData,
            _u: PhantomData,
        }
    }

    /// Constructs a new `Tree` from a given root `Cluster` and dataset.
    pub fn from_root_and_data(root: C, data: D) -> Self {
        let depth = root.max_leaf_depth();
        Self {
            data,
            root,
            depth,
            _i: PhantomData,
            _u: PhantomData,
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
    pub fn partition<P: PartitionCriterion<U>>(mut self, criteria: &P, seed: Option<u64>) -> Self {
        self.root = self.root.partition(&mut self.data, criteria, seed);
        self.depth = self.root.max_leaf_depth();
        self
    }

    /// Returns the `Cluster` with the given `offset` and `cardinality`.
    ///
    /// # Arguments
    ///
    /// * `offset`: The offset of the `Cluster` to return.
    /// * `cardinality`: The cardinality of the `Cluster` to return.
    ///
    /// # Returns
    ///
    /// The `Cluster` with the given `offset` and `cardinality` if it exists.
    /// Otherwise, `None`.
    pub fn get_cluster(&self, offset: usize, cardinality: usize) -> Option<&C> {
        self.root.descend_to(offset, cardinality)
    }

    /// Returns a reference to the data used to build the `Tree`.
    pub const fn data(&self) -> &D {
        &self.data
    }

    /// The cardinality of the `Tree`, i.e. the number of instances in the data.
    pub fn cardinality(&self) -> usize {
        self.root.cardinality()
    }

    /// The radius of the root of the `Tree`.
    pub fn radius(&self) -> U {
        self.root.radius()
    }

    /// The root `Cluster` of the `Tree`.
    pub const fn root(&self) -> &C {
        &self.root
    }

    /// The depth of the `Tree`.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Saves a tree to a given location
    ///
    /// The path given will point to a newly created folder which will
    /// store all necessary data for tree reconstruction.
    ///
    /// The directory structure looks like the following:
    ///
    /// ```text
    /// /user/given/path/
    ///    |- dataset      <-- The serialized dataset.
    ///    |- clusters     <-- Clusters are serialized to a single file.
    /// ```
    ///
    /// # Arguments
    ///
    /// * `path` - The path to save the tree to.
    ///
    /// # Errors
    ///
    /// * If `path` does not exist.
    /// * If `path` cannot be written to.
    /// * If there are any serialization errors with the dataset.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if !path.exists() {
            return Err("Given path does not exist".to_string());
        }

        let dataset_path = path.join("dataset");
        self.data.save(&dataset_path)?;

        let cluster_path = path.join("clusters");
        self.root.save(&cluster_path)?;

        Ok(())
    }

    /// Reconstructs a `Tree` from a directory `path` with associated metric `metric`. Returns the
    /// reconstructed tree.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load the tree from.
    /// * `metric` - The metric to use for the tree.
    /// * `is_expensive` - Whether or not the metric is expensive to compute.
    ///
    /// # Returns
    ///
    /// The reconstructed tree.
    ///
    /// # Errors
    ///
    /// * If `path` does not exist.
    /// * If `path` does not contain a valid tree. See `save` for more information
    /// on the directory structure.
    /// * If the `path` cannot be read from.
    /// * If there are any deserialization errors with the dataset.
    /// * If there are any deserialization errors with the clusters.
    pub fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String> {
        if !path.exists() {
            return Err("Given path does not exist".to_string());
        }

        // Aliases to relevant paths
        let cluster_path = path.join("clusters");
        let dataset_path = path.join("dataset");

        if !(cluster_path.exists() && dataset_path.exists()) {
            return Err("Saved tree is malformed".to_string());
        }

        let data = D::load(&dataset_path, metric, is_expensive)?;
        let root = C::load(&cluster_path)?;

        Ok(Self {
            data,
            depth: root.max_leaf_depth(),
            root,
            _i: PhantomData,
            _u: PhantomData,
        })
    }
}
