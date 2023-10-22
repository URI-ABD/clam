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
    /// store all necessary data for tree reconstruction.
    ///
    /// The directory structure looks like the following:
    /// ```text
    /// user_given_path/
    ///      clusters/   <-- Clusters are serialized using their hex name.
    ///      childinfo/  <-- Information about clusters immediate children.
    ///      dataset/    <-- The serialized dataset.
    ///      leaves.json <-- A json file of leaf names.
    /// ````
    ///
    /// # Errors
    /// Errors out on any directory or file creation issues
    #[allow(clippy::missing_panics_doc)]
    pub fn save(&self, path: &Path) -> Result<(), String> {
        // Create our directory if needed
        let dirbuilder = DirBuilder::new();

        if !path.exists() {
            dirbuilder.create(path).map_err(|e| e.to_string())?;
        }

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

    /// Reconstructs a `Tree` from a directory `path` with associated metric `metric`. Returns the
    /// reconstructed tree.
    ///
    /// # Errors
    /// This function will return an error if there's any file creation, file reading, or json deserialization issues.
    #[allow(clippy::missing_panics_doc)]
    pub fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String> {
        // Alises to relevant directories
        let cluster_path = path.join("clusters");
        let childinfo_path = path.join("childinfo");
        let dataset_dir = path.join("dataset").join("data");
        let dataset = D::load(&dataset_dir, metric, is_expensive)?;

        // Load the root in
        let root = recover_serialized_cluster(cluster_path.join("1"))?;

        // Open up and read the names of leaf indices
        let mut handle = File::open(path.join("leaves.json")).map_err(|e| e.to_string())?;
        let mut leaf_buf = String::new();
        handle.read_to_string(&mut leaf_buf).map_err(|e| e.to_string())?;

        let leaf_names: Vec<String> = serde_json::from_str(&leaf_buf).map_err(|e| e.to_string())?;
        let mut boxed_root = Box::new(root);

        // Now, for each leaf, we build out the tree up to that leaf
        for leaf in leaf_names {
            let mut cur = &mut boxed_root;
            let leaf_history = Cluster::<U>::name_to_history(&leaf);

            // We start from index 1 to skip the identically 1 prefix at index 0
            for step in 1..leaf_history.len() {
                let branch = leaf_history[step];

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
                    let left: Cluster<U> = recover_serialized_cluster(cluster_path.join(left_name))?;
                    let right: Cluster<U> = recover_serialized_cluster(cluster_path.join(right_name))?;

                    // Get the childinfo (arg_l, arg_r, etc.)
                    let parent_name = Cluster::<U>::history_to_name(&leaf_history[0..step]);
                    let childinfo = recover_serialized_childinfo(childinfo_path.join(&parent_name))?;

                    // Reconstruct the children
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

                // Choose which branch to take
                if branch {
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

/// Serializes a serializeable (`impl Serialize`) object to a given path
///
/// # Errors
/// This function will error out on serialization or file i/o errors.
fn serialize_to_file<S: Serialize>(path: PathBuf, object: &S) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| e.to_string())?;
    let info_string = serde_json::to_string(&object).map_err(|e| e.to_string())?;
    file.write_all(info_string.as_bytes()).map_err(|e| e.to_string())
}

/// Recovers a `Cluster` from a serialized cluster contained in a given file. Does not recover child info
///
/// # Errors
/// This function will error out on any deserialization or file i/o errors.
fn recover_serialized_cluster<U: Number>(path: PathBuf) -> Result<Cluster<U>, String> {
    let mut buffer = String::new();
    let mut cluster_handle = File::open(path).map_err(|e| e.to_string())?;

    cluster_handle.read_to_string(&mut buffer).map_err(|e| e.to_string())?;

    let cluster: SerializedCluster = serde_json::from_str(&buffer).map_err(|e| e.to_string())?;
    Ok(cluster.into_partial_cluster())
}

/// Recovers child information for a given cluster
///
/// # Errors
/// This function will error out on any deserialization or file i/o errors.
fn recover_serialized_childinfo(path: PathBuf) -> Result<SerializedChildren, String> {
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
    use distances::Number;
    use tempdir::TempDir;

    use crate::{
        core::cluster::{Cluster, _cluster::Children},
        Dataset, Instance, PartitionCriteria, Tree, VecDataset,
    };

    fn metric(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        distances::vectors::euclidean(x, y)
    }

    //I: Instance, U: Number, D: Dataset<I, U>
    fn assert_trees_equal<I: Instance, U: Number>(
        tree1: &Tree<I, U, VecDataset<I, U>>,
        tree2: &Tree<I, U, VecDataset<I, U>>,
    ) {
        // TODO: (OWM) Right now tree depths are never actually recalculated after partitioning, so `depth` is always
        // zero on a normal tree. This is not the case with a recovered tree.
        // assert_eq!(tree1.depth, tree2.depth, "Tree depths inequal");
        assert_clusters_equal(&tree1.root, &tree1.data, &tree2.root, &tree2.data);
    }

    fn assert_clusters_equal<I: Instance, U: Number>(
        cluster1: &Cluster<U>,
        dataset1: &VecDataset<I, U>,
        cluster2: &Cluster<U>,
        dataset2: &VecDataset<I, U>,
    ) {
        // Assert their cardinalities
        assert_eq!(cluster1.cardinality, cluster2.cardinality);

        // Resolve centers
        let (center1, center2) = (&dataset1.data[cluster1.arg_center], &dataset2.data[cluster2.arg_center]);
        let (radial1, radial2) = (&dataset1.data[cluster1.arg_radial], &dataset2.data[cluster2.arg_radial]);

        // Metric is assumed to be shared (Good idea?)
        let metric = dataset1.metric();

        // Assert centers and radials are equal
        assert_eq!(metric(center1, center2), U::zero());
        assert_eq!(metric(radial1, radial2), U::zero());

        // Get children and assert they are of equal optionality
        let (children1, children2) = (&cluster1.children, &cluster2.children);

        assert!(
            children1.is_none() && children2.is_none() || children1.is_some() && children2.is_some(),
            "One cluster has children, the other does not"
        );

        // If we have children, assert their relevant details are equal and then recurse on their clusters
        if children1.is_some() {
            let Children {
                left: left_1,
                right: right_1,
                arg_l: arg_l_1,
                arg_r: arg_r_1,
                ..
            } = children1.as_ref().unwrap();

            let Children {
                left: left_2,
                right: right_2,
                arg_l: arg_l_2,
                arg_r: arg_r_2,
                ..
            } = children2.as_ref().unwrap();

            let (l_1, l_2) = (&dataset1.data[*arg_l_1], &dataset1.data[*arg_l_2]);
            let (r_1, r_2) = (&dataset1.data[*arg_r_1], &dataset1.data[*arg_r_2]);

            assert_eq!(metric(l_1, l_2), U::zero());
            assert_eq!(metric(r_1, r_2), U::zero());

            assert_clusters_equal(left_1, dataset1, left_2, dataset2);
            assert_clusters_equal(right_1, dataset1, right_2, dataset2);
        }
    }

    #[test]
    fn recover_tiny() {
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

        // Generate some tree from a small dataset
        let name = "test".to_string();
        let data = VecDataset::new(name, data, metric, false);
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
        let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria);

        let tree_path = TempDir::new("tree_tiny").unwrap();

        // Save the tree
        raw_tree.save(&tree_path.path()).unwrap();

        // Recover the tree
        let recovered_tree = Tree::<Vec<f32>, f32, VecDataset<_, _>>::load(&tree_path.path(), metric, false).unwrap();

        // Assert recovering was successful
        assert_trees_equal(&raw_tree, &recovered_tree);
    }

    #[test]
    fn recover_medium() {
        let (dimensionality, min_val, max_val) = (10, -1., 1.);
        let seed = 42;

        let data = symagen::random_data::random_f32(10_000, dimensionality, min_val, max_val, seed);
        let name = "test".to_string();
        let data = VecDataset::<_, f32>::new(name, data, metric, false);
        let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::new(true).with_min_cardinality(1);

        let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria);

        let tree_path = TempDir::new("tree_medium").unwrap();

        // Save the tree
        raw_tree.save(&tree_path.path()).unwrap();

        // Recover the tree
        let recovered_tree = Tree::<Vec<f32>, f32, VecDataset<_, _>>::load(&tree_path.path(), metric, false).unwrap();

        // Assert recovering was successful
        assert_trees_equal(&raw_tree, &recovered_tree);
    }

    #[test]
    fn recover_large() {
        let (dimensionality, min_val, max_val) = (10, -1., 1.);
        let seed = 42;

        let data = symagen::random_data::random_f32(100_000, dimensionality, min_val, max_val, seed);
        let name = "test".to_string();
        let data = VecDataset::<_, f32>::new(name, data, metric, false);
        let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::new(true).with_min_cardinality(1);

        let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria);

        let tree_path = TempDir::new("tree_large").unwrap();

        // Save the tree
        raw_tree.save(&tree_path.path()).unwrap();

        // Recover the tree
        let recovered_tree = Tree::<Vec<f32>, f32, VecDataset<_, _>>::load(&tree_path.path(), metric, false).unwrap();

        // Assert recovering was successful
        assert_trees_equal(&raw_tree, &recovered_tree);
    }
}
