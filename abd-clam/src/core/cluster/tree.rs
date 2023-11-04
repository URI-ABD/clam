//! A `Tree` represents a hierarchy of "similar" instances from a metric-`Space`.

use core::marker::PhantomData;
use std::{
    fs::{create_dir, File},
    io::{Read, Write},
    path::Path,
};

use super::{
    SerializedCluster,
    _cluster::{Children, SerializedChildren},
};
use crate::{utils, Cluster, Dataset, Instance, PartitionCriteria};
use distances::Number;
use serde::{Deserialize, Serialize};

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
    _i: PhantomData<I>,
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
            _i: PhantomData,
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
        self.depth = self.root.max_leaf_depth();
        self
    }

    /// Sets the `Cluster` ratios for anomaly detection and related applications.
    ///
    /// This should only be called on the root `Cluster` after calling `partition`.
    ///
    /// # Arguments
    ///
    /// * `normalized`: Whether to apply Gaussian error normalization to the ratios.
    ///
    /// # Panics
    ///
    /// * If this method is called on a non-root `Cluster`.
    /// * If this method is called before calling `partition` on the root `Cluster`.
    #[must_use]
    pub fn with_ratios(mut self, normalize: bool) -> Self {
        // TODO: Let's move this to a method on Cluster.

        self.root = self.root.set_child_parent_ratios([1.0; 6]);

        if normalize {
            let all_ratios = self
                .root
                .subtree()
                .iter()
                .map(|c| {
                    c.ratios()
                        .unwrap_or_else(|| unreachable!("We just set the ratios above."))
                })
                .collect::<Vec<_>>();

            let all_ratios = utils::rows_to_cols(&all_ratios);

            // mean of each column
            let means: [f64; 6] = utils::calc_row_means(&all_ratios);

            // sd of each column
            let sds: [f64; 6] = utils::calc_row_sds(&all_ratios);

            self.root.set_normalized_ratios(means, sds);
        }

        self
    }

    /// Returns a reference to the data used to build the `Tree`.
    pub const fn data(&self) -> &D {
        &self.data
    }

    /// The cardinality of the `Tree`, i.e. the number of instances in the data.
    pub const fn cardinality(&self) -> usize {
        self.root.cardinality()
    }

    /// The radius of the root of the `Tree`.
    pub const fn radius(&self) -> U {
        self.root.radius()
    }

    /// The root `Cluster` of the `Tree`.
    pub const fn root(&self) -> &Cluster<U> {
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
    ///    |- clusters/    <-- Clusters are serialized using their hex name.
    ///    |- childinfo/   <-- Information about clusters immediate children.
    ///    |- dataset/     <-- The serialized dataset.
    ///    |- leaves.json  <-- A json file of leaf names.
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

        // Create cluster directory
        let cluster_path = path.join("clusters");
        create_dir(&cluster_path).map_err(|e| e.to_string())?;

        // Create childinfo directory
        let childinfo_path = path.join("childinfo");
        create_dir(&childinfo_path).map_err(|e| e.to_string())?;

        // Create dataset directory
        let dataset_dir = path.join("dataset");
        create_dir(&dataset_dir).map_err(|e| e.to_string())?;

        // List of leaf clusters (Used for loading in the tree later)
        let mut leaves: Vec<String> = vec![];

        // Traverse the tree, serializing each cluster
        let mut stack = vec![&self.root];
        while let Some(cur) = stack.pop() {
            let filename: String = cur.name();

            match cur.children() {
                Some([left, right]) => {
                    stack.push(left);
                    stack.push(right);
                }
                None => {
                    leaves.push(filename.clone());
                }
            }

            // Write out the serialized cluster
            let (serialized, children) = SerializedCluster::from_cluster(cur);
            let node_path = cluster_path.join(&filename);
            serialize_to_file(&node_path, &serialized)?;

            if let Some(childinfo) = children {
                let info_path = childinfo_path.join(&filename);
                serialize_to_file(&info_path, &childinfo)?;
            }
        }

        // Save the dataset
        let saved_dataset_path = dataset_dir.join("data");
        self.data.save(&saved_dataset_path)?;

        // Save the leaf data
        let leaf_data_path = path.join("leaves.json");
        serialize_to_file(&leaf_data_path, &leaves)?;

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

        // Alises to relevant directories
        let cluster_path = path.join("clusters");
        let childinfo_path = path.join("childinfo");
        let dataset_dir = path.join("dataset").join("data");

        if !(cluster_path.exists() && childinfo_path.exists() && dataset_dir.exists()) {
            return Err("Save directory is malformed".to_string());
        }

        let dataset = D::load(&dataset_dir, metric, is_expensive)?;

        // Load the root in
        let root = recover_serialized_cluster(&cluster_path.join("1"))?;

        // Open up and read the names of leaf indices
        let leaf_names: Vec<String> = deserialize_from_file(&path.join("leaves.json"), &mut Vec::new())?;
        let mut boxed_root = Box::new(root);

        // Now, for each leaf, we build out the tree up to that leaf
        for leaf in leaf_names {
            let mut cur = &mut boxed_root;
            let leaf_history = Cluster::<U>::name_to_history(&leaf);

            // We start from index 1 to skip the identically 1 prefix at index 0
            for step in 1..leaf_history.len() {
                let go_right = leaf_history[step];

                // TODO: Child reconstruction should be moved to a method on Cluster

                // If we don't have any children here, we need to build them out
                if cur.children.is_none() {
                    // Construct the names for the left and right children
                    let mut left_history = leaf_history[0..step].to_vec();
                    left_history.push(false);

                    let mut right_history = leaf_history[0..step].to_vec();
                    right_history.push(true);

                    let left_name = Cluster::<U>::history_to_name(&left_history);
                    let right_name = Cluster::<U>::history_to_name(&right_history);

                    // Deserialize the left and right child clusters
                    let left: Cluster<U> = recover_serialized_cluster(&cluster_path.join(left_name))?;
                    let right: Cluster<U> = recover_serialized_cluster(&cluster_path.join(right_name))?;

                    // Get the childinfo (arg_l, arg_r, etc.)
                    let parent_name = Cluster::<U>::history_to_name(&leaf_history[0..step]);
                    let childinfo = recover_serialized_childinfo(&childinfo_path.join(&parent_name))?;

                    // Reconstruct the children
                    cur.children = Some(Children {
                        left: Box::new(left),
                        right: Box::new(right),
                        arg_l: childinfo.arg_l,
                        arg_r: childinfo.arg_r,
                        polar_distance: <U as Number>::from_le_bytes(&childinfo.polar_distance_bytes),
                    });
                }

                let children = cur
                    .children
                    .as_mut()
                    .unwrap_or_else(|| unreachable!("We have already checked if `children` is None."));

                // Choose which branch to take
                if go_right {
                    cur = &mut children.right;
                } else {
                    cur = &mut children.left;
                }
            }
        }

        let root = *boxed_root;

        Ok(Self {
            data: dataset,
            depth: root.max_leaf_depth(),
            root,
            _i: PhantomData,
        })
    }
}

/// Serializes an object to a given path.
///
/// # Arguments
///
/// * `path` - The path to serialize the object to.
/// * `object` - The object to serialize.
///
/// # Errors
///
/// * If `path` cannot be written to.
/// * If there are any serialization errors with the object.
fn serialize_to_file<S: Serialize>(path: &Path, object: &S) -> Result<(), String> {
    // TODO: At some point we will encode `object` as bytes instead of json
    let mut file = File::create(path).map_err(|e| e.to_string())?;
    let object = postcard::to_allocvec(object).map_err(|e| e.to_string())?;
    file.write_all(&object).map_err(|e| e.to_string())
}

/// Deserializes an object from a given path.
///
/// # Arguments
///
/// * `path` - The path to deserialize the object from.
///
/// # Errors
///
/// * If `path` cannot be read from.
/// * If there are any deserialization errors with the object.
fn deserialize_from_file<'a, D: Deserialize<'a>>(path: &Path, buffer: &'a mut Vec<u8>) -> Result<D, String> {
    let mut handle = File::open(path).map_err(|e| e.to_string())?;
    handle.read_to_end(buffer).map_err(|e| e.to_string())?;
    postcard::from_bytes(buffer).map_err(|e| e.to_string())
}

/// Recovers a `Cluster` from a serialized cluster contained in a given file. Does not recover child info
///
/// # Errors
/// This function will error out on any deserialization or file i/o errors.
fn recover_serialized_cluster<U: Number>(path: &Path) -> Result<Cluster<U>, String> {
    let cluster: SerializedCluster = deserialize_from_file(path, &mut Vec::new())?;
    Ok(cluster.into_partial_cluster())
}

/// Recovers child information for a given cluster
///
/// # Errors
/// This function will error out on any deserialization or file i/o errors.
fn recover_serialized_childinfo(path: &Path) -> Result<SerializedChildren, String> {
    deserialize_from_file(path, &mut Vec::new())
}
