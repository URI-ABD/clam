//! A `Tree` represents a hierarchy of "similar" instances from a metric-`Space`.

use core::marker::PhantomData;
use std::{
    fs::{DirBuilder, File},
    io::Write,
    path::Path,
};

use distances::Number;

use crate::{Cluster, Dataset, Instance, PartitionCriteria};

use super::SerializedCluster;

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
        //      clusters/ <-- Clusters are serialized using their hex name.
        //                     Leaves have their name prepended with 'l_'
        //      dataset/

        // Create our directory
        let dirbuilder = DirBuilder::new();
        dirbuilder.create(path).map_err(|e| e.to_string())?;

        // Create cluster directory
        let cluster_path = path.join("clusters");
        dirbuilder.create(cluster_path).map_err(|e| e.to_string())?;

        // Traverse the tree, serializing each cluster
        let mut stack = vec![&self.root];
        while let Some(cur) = stack.pop() {
            // Our filename is dynamic based on if the cluster is a leaf or not
            let mut filename: String = String::new();

            // If the cluster is a parent, we push the children to the queue
            if !cur.is_leaf() {
                // Unwrapping is justified here because we validated that the cluster
                // is a leaf before reaching this code
                #[allow(clippy::unwrap_used)]
                let [l, r] = cur.children().unwrap();
                stack.push(l);
                stack.push(r);

                // Append "l_" to filename if the node is a leaf
                filename += "l_";
            }

            // Finalize the filename
            filename += &cur.name();

            // Create the path to and open the serialized cluster file
            let node_path = path.join(filename);
            let mut file = File::create(node_path).map_err(|e| e.to_string())?;

            // Write out the serialized cluster
            let serialized = SerializedCluster::from_cluster(cur);
            let serialized = serde_json::to_string(&serialized).map_err(|e| e.to_string())?;
            write!(file, "{serialized}").map_err(|e| e.to_string())?;
        }

        // Serialize our dataset
        let dataset_path = path.join("dataset");
        dirbuilder.create(dataset_path).map_err(|e| e.to_string())?;
        self.data.save(path)?;

        Ok(())
    }

    /// # Errors
    pub fn load(_path: &Path) -> Result<Self, String> {
        // Load the dataset in

        // Load the root in

        // Load the leaves into a vec

        // for each leaf { build out to that leaf }
        todo!()
    }
}
