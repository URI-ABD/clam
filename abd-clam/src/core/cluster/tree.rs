//! A `Tree` represents a hierarchy of "similar" instances from a metric-`Space`.

use core::marker::PhantomData;
use std::{
    fs::{DirBuilder, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use super::{SerializedChildren, SerializedCluster, _cluster::Children};
use crate::{Cluster, Dataset, Instance, PartitionCriteria};
use distances::Number;
use serde::Serialize;

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

    /// Saves a tree to a given location
    ///
    /// The path given will point to a newly created folder which will
    /// store all necessary data for reconstruction.
    ///
    /// # Errors
    /// .
    #[allow(clippy::missing_panics_doc)]
    pub fn save(&self, path: &Path) -> Result<(), String> {
        // General structure
        // user_given_path/
        //      clusters/   <-- Clusters are serialized using their hex name.
        //      childinfo/  <-- Information about clusters immediate children.
        //      dataset/    <-- The serialized dataset.
        //      leaves.json <-- A json file of leaf names.
        // Create our directory
        let dirbuilder = DirBuilder::new();
        dirbuilder.create(path).map_err(|e| e.to_string())?;

        // Create cluster directory
        let cluster_path = path.join("clusters");
        dirbuilder.create(&cluster_path).map_err(|e| e.to_string())?;

        // Create childinfo directory
        let childinfo_path = path.join("childinfo");
        dirbuilder.create(&childinfo_path).map_err(|e| e.to_string())?;

        // Create dataset directory
        let dataset_dir = path.join("dataset");
        dirbuilder.create(&dataset_dir).map_err(|e| e.to_string())?;

        // List of leaf clusters (Used for loading in the tree later)
        let mut leaves: Vec<String> = vec![];

        // Traverse the tree, serializing each cluster
        let mut stack = vec![&self.root];
        while let Some(cur) = stack.pop() {
            let filename: String = cur.name();

            // If the cluster is a parent, we push the children to the queue
            if cur.is_leaf() {
                leaves.push(filename.clone());
            } else {
                // Unwrapping is justified here because we validated that the cluster
                // is a leaf before reaching this code
                #[allow(clippy::unwrap_used)]
                let [l, r] = cur.children().unwrap();
                stack.push(l);
                stack.push(r);
            }

            // Write out the serialized cluster
            let (serialized, children) = SerializedCluster::from_cluster(cur);
            let node_path = cluster_path.join(&filename);
            serialize_to_file(node_path, &serialized)?;

            if let Some(childinfo) = children {
                let info_path = childinfo_path.join(&filename);
                serialize_to_file(info_path, &childinfo)?;
            }
        }

        // Save the dataset
        let saved_dataset_path = dataset_dir.join("data");
        self.data.save(&saved_dataset_path)?;

        // Save the leaf data
        let leaf_data_path = path.join("leaves.json");
        serialize_to_file(leaf_data_path, &leaves)?;

        Ok(())
    }

    /// # Errors
    #[allow(clippy::missing_panics_doc)]
    pub fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String> {
        let cluster_path = path.join("clusters");
        let childinfo_path = path.join("childinfo");
        let dataset_dir = path.join("dataset").join("data");
        let dataset = D::load(&dataset_dir, metric, is_expensive)?;

        // Load the root in
        let root = recover_serialized_cluster(cluster_path.join("1"))?;

        // Leaf list
        let mut handle = File::open(path.join("leaves.json")).map_err(|e| e.to_string())?;
        let mut leaf_buf = String::new();
        handle.read_to_string(&mut leaf_buf).map_err(|e| e.to_string())?;

        let leaf_names: Vec<String> = serde_json::from_str(&leaf_buf).map_err(|e| e.to_string())?;

        let mut boxed_root = Box::new(root);

        // Now, for each leaf, we build out the tree up to that leaf
        // This is bounded O(nd) where n is the # of nodes, d is the depth of the tree
        for leaf in leaf_names {
            let mut cur = &mut boxed_root;
            let leaf_history = Cluster::<U>::name_to_history(&leaf);

            for step in 1..leaf_history.len() {
                let branch = leaf_history[step];

                if cur.children.is_none() {
                    let mut left_history = leaf_history[0..step].to_vec();
                    left_history.push(false);

                    let left_name = Cluster::<U>::history_to_name(&left_history);
                    let left: Cluster<U> = recover_serialized_cluster(cluster_path.join(left_name))?;

                    let mut right_history = leaf_history[0..step].to_vec();
                    right_history.push(true);

                    let right_name = Cluster::<U>::history_to_name(&right_history);
                    let right: Cluster<U> = recover_serialized_cluster(cluster_path.join(right_name))?;

                    let parent_name = Cluster::<U>::history_to_name(&leaf_history[0..step]);
                    let childinfo = recover_childinfo(childinfo_path.join(&parent_name))?;

                    cur.children = Some(Children {
                        left: Box::new(left),
                        right: Box::new(right),
                        arg_l: childinfo.arg_l,
                        arg_r: childinfo.arg_r,
                        polar_distance: <U as Number>::from_le_bytes(&childinfo.polar_distance_bytes),
                    });
                }

                // Unwrap is justified here because we always check if cur.children is None and
                // load in the children if it is.
                #[allow(clippy::unwrap_used)]
                let children = cur.children.as_mut().unwrap();

                if branch {
                    cur = &mut children.right;
                } else {
                    cur = &mut children.left;
                }
            }
        }

        // TODO: Fix depth
        Ok(Self {
            data: dataset,
            root: *boxed_root,
            depth: 0,
            _i: PhantomData,
        })
    }
}

///
fn serialize_to_file<S: Serialize>(path: PathBuf, object: &S) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| e.to_string())?;
    let info_string = serde_json::to_string(&object).map_err(|e| e.to_string())?;
    file.write_all(info_string.as_bytes()).map_err(|e| e.to_string())
}

///
fn recover_serialized_cluster<U: Number>(path: PathBuf) -> Result<Cluster<U>, String> {
    let mut buffer = String::new();
    let mut cluster_handle = File::open(path).map_err(|e| e.to_string())?;

    cluster_handle.read_to_string(&mut buffer).map_err(|e| e.to_string())?;

    let cluster: SerializedCluster = serde_json::from_str(&buffer).map_err(|e| e.to_string())?;
    Ok(cluster.into_partial_cluster())
}

///
fn recover_childinfo(path: PathBuf) -> Result<SerializedChildren, String> {
    let mut buffer = String::new();
    let mut childinfo_handle = File::open(path).map_err(|e| e.to_string())?;

    childinfo_handle
        .read_to_string(&mut buffer)
        .map_err(|e| e.to_string())?;

    let childinfo: SerializedChildren = serde_json::from_str(&buffer).map_err(|e| e.to_string())?;
    Ok(childinfo)
}

#[cfg(test)]
mod tests {
    use std::env::temp_dir;

    use crate::{PartitionCriteria, Tree, VecDataset};

    fn metric(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        distances::vectors::euclidean(x, y)
    }

    #[test]
    fn directory_structure_after_save() {
        let data = vec![
            vec![10.],
            vec![1.],
            vec![-5.],
            vec![8.],
            vec![3.],
            vec![2.],
            vec![0.5],
            vec![0.],
        ];
        let name = "test".to_string();
        let data = VecDataset::new(name, data, metric, false);
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

        let tree = Tree::new(data, Some(42)).partition(&partition_criteria);

        let leaf_indices = tree.root.indices().collect::<Vec<_>>();
        let tree_indices = (0..tree.root.cardinality).collect::<Vec<_>>();

        assert_eq!(leaf_indices, tree_indices);

        let tree_path = temp_dir().join("tree");

        if tree_path.exists() {
            std::fs::remove_dir_all(&tree_path).unwrap();
        }

        tree.save(&tree_path).unwrap();

        let new_tree = Tree::<Vec<f32>, f32, VecDataset<_, _>>::load(&tree_path, metric, false).unwrap();
        dbg!(new_tree);
    }
}
