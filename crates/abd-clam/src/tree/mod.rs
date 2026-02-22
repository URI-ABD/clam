//! A `Tree` of `Clusters` for use in CLAM.

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::DistanceValue;

mod cluster;
mod dataset;
mod partition;

pub use cluster::Cluster;
pub use dataset::Dataset;
pub use partition::strategy::{self as partition_strategy, PartitionStrategy};

// TODO(Najib): Add methods to annotate clusters after tree creation.

/// The `Tree` struct is the main data structure used in CLAM.
///
/// If contains the root `Cluster`, the items stored in it, and the metric used to compute distances between items.
///
/// # Type Parameters
///
/// - `D`: The type of the dataset used in the tree.
/// - `M`: The type of the metric function used to compute distances between items from the dataset.
/// - `T`: The type of the distance values used in the tree.
/// - `A`: The type of any annotations that can be added to clusters.
#[must_use]
#[derive(Clone, Debug)]
pub struct Tree<D, M, T, A> {
    /// The dataset used in the tree.
    pub(crate) dataset: D,
    /// The metric used to compute distances between items.
    pub(crate) metric: M,
    /// All clusters in the tree. This is a mapping from `cluster.center_index` to `cluster`.
    pub(crate) cluster_map: HashMap<usize, Cluster<T, A>>,
}

/// Minimal constructors for `Tree`.
///
/// - The identifier type is set to `usize` and will be the index of the item in the original vector.
/// - The annotation type is set to `()`, meaning that no annotations are stored in the tree.
/// - The default [`PartitionStrategy`](PartitionStrategy) is used to build a binary tree.
impl<D, T, M> Tree<D, M, T, ()>
where
    D: Dataset,
    M: Fn(&D::Item, &D::Item) -> T,
    T: DistanceValue,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn new_minimal(dataset: D, metric: M) -> Result<Self, &'static str> {
        if dataset.is_empty() {
            return Err("Cannot create a Tree with an empty dataset.");
        }

        Self::new(dataset, metric, &PartitionStrategy::default(), &|_| ())
    }

    /// Parallel version of [`Self::new_minimal`].
    ///
    /// # Errors
    ///
    /// See [`Self::new_minimal`].
    pub fn par_new_minimal(dataset: D, metric: M) -> Result<Self, &'static str>
    where
        D: Send + Sync,
        D::Id: Send + Sync,
        D::Item: Send + Sync,
        M: Send + Sync,
        T: Send + Sync,
    {
        if dataset.is_empty() {
            return Err("Cannot create a Tree with an empty dataset.");
        }

        Self::par_new(dataset, metric, &PartitionStrategy::default(), &|_| ())
    }
}

/// Various getter methods for `Tree`.
impl<D, M, T, A> Tree<D, M, T, A> {
    /// Provides ownership of the members of the `Tree`.
    pub fn into_parts(self) -> (D, M, HashMap<usize, Cluster<T, A>>) {
        (self.dataset, self.metric, self.cluster_map)
    }

    /// Creates a `Tree` from its members.
    pub(crate) const fn from_parts(dataset: D, metric: M, cluster_map: HashMap<usize, Cluster<T, A>>) -> Self {
        Self { dataset, metric, cluster_map }
    }

    /// Returns a reference to the dataset used in the tree.
    pub const fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Returns a reference to the metric used in the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

    /// Returns a reference to the hash map of all clusters in the tree.
    pub const fn cluster_map(&self) -> &HashMap<usize, Cluster<T, A>> {
        &self.cluster_map
    }

    /// Returns a reference to a cluster in the tree given its center index, if it exists.
    ///
    /// # Errors
    ///
    /// If no cluster with the given center index exists in the tree.
    pub fn get_cluster(&self, id: usize) -> Result<&Cluster<T, A>, String> {
        self.cluster_map
            .get(&id)
            .ok_or_else(|| format!("No cluster with center index {id} found in the tree."))
    }

    /// Returns a mutable reference to a cluster in the tree given its center index, if it exists.
    ///
    /// # Errors
    ///
    /// If no cluster with the given center index exists in the tree.
    pub fn get_cluster_mut(&mut self, id: usize) -> Result<&mut Cluster<T, A>, String> {
        self.cluster_map
            .get_mut(&id)
            .ok_or_else(|| format!("No cluster with center index {id} found in the tree."))
    }

    /// Returns the number of clusters in the tree.
    pub fn n_clusters(&self) -> usize {
        self.cluster_map.len()
    }

    /// Returns all clusters in the tree.
    pub fn all_clusters(&self) -> Vec<&Cluster<T, A>> {
        self.cluster_map.values().collect()
    }

    /// Returns all clusters in the tree in sorted order of their center indices, i.e. in pre-order traversal over the tree.
    pub fn sorted_clusters(&self) -> Vec<&Cluster<T, A>> {
        let mut clusters = self.cluster_map.values().collect::<Vec<_>>();
        clusters.sort_by_key(|c| c.center_index());
        clusters
    }

    /// Returns a reference to the root cluster of the tree.
    pub fn root(&self) -> &Cluster<T, A> {
        self.cluster_map
            .get(&0)
            .unwrap_or_else(|| unreachable!("Tree must have a root cluster with center_index 0"))
    }

    /// Returns references to the children of the given cluster, if any.
    pub fn children_of(&self, cluster: &Cluster<T, A>) -> Option<Vec<&Cluster<T, A>>> {
        cluster
            .child_center_indices()
            .map(|center_indices| center_indices.iter().filter_map(|&ci| self.cluster_map.get(&ci)).collect())
    }
}

/// Various setters for `Tree`.
impl<D, M, T, A> Tree<D, M, T, A> {
    /// Changes the metric used in the tree to the provided one.
    pub fn with_metric<NewM>(self, metric: NewM) -> Tree<D, NewM, T, A> {
        Tree {
            dataset: self.dataset,
            metric,
            cluster_map: self.cluster_map,
        }
    }

    /// Applies the given closure to every item and identifier.
    pub fn apply_to_dataset<F, NewD>(self, f: &F) -> Tree<NewD, M, T, A>
    where
        F: Fn(D) -> NewD,
    {
        Tree {
            dataset: f(self.dataset),
            metric: self.metric,
            cluster_map: self.cluster_map,
        }
    }
}

/// Constructors for `Tree`.
impl<D, M, T, A> Tree<D, M, T, A>
where
    D: Dataset,
    M: Fn(&D::Item, &D::Item) -> T,
    T: DistanceValue,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Arguments
    ///
    /// * `dataset` - A collection of items and their identifiers.
    /// * `metric` - A function that computes the distance between two items.
    /// * `strategy` - A `PartitionStrategy` that defines how to partition clusters.
    /// * `annotator` - A function that annotates clusters as they are created.
    ///
    /// # Errors
    ///
    /// If `dataset` is empty.
    pub fn new<P, Ann>(mut dataset: D, metric: M, strategy: &PartitionStrategy<P>, annotator: &Ann) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool,
        Ann: Fn(&Cluster<T, A>) -> A,
    {
        if dataset.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items", dataset.cardinality());

        let mut cluster_map = HashMap::new();

        // The `frontier` holds clusters that were just created but whose children have not yet been created.
        let mut frontier = vec![Cluster::new(dataset.as_mut_slice(), &metric, strategy)];
        ftlog::info!("Created root cluster with cardinality {}", frontier[0].0.cardinality);

        while let Some((mut cluster, splits)) = frontier.pop() {
            // For each split, create the child cluster and get the splits for its children and add them to the frontier.
            frontier.extend(splits.into_iter().map(|(offset, child_items)| {
                // Create child cluster and get items for its children.
                let (mut child, mut child_splits) = Cluster::new(child_items, &metric, strategy);

                ftlog::info!(
                    "Created a child cluster with cardinality {}, parent center index {}",
                    child.cardinality,
                    cluster.center_index
                );

                // Increment relevant indices
                child.depth = cluster.depth + 1;
                child.increment_indices(offset);
                for (ci, _) in &mut child_splits {
                    *ci += offset;
                }

                // Set parent center index for child cluster.
                child.parent_center_index = Some(cluster.center_index);
                (child, child_splits)
            }));

            // Annotate cluster and insert into map.
            cluster.annotation = annotator(&cluster);
            cluster_map.insert(cluster.center_index, cluster);
        }

        ftlog::info!("Finished creating tree with {} items", dataset.cardinality());
        Ok(Self { dataset, metric, cluster_map })
    }
}

/// Parallelized constructors for `Tree`.
impl<D, M, T, A> Tree<D, M, T, A>
where
    D: Dataset + Send + Sync,
    D::Id: Send + Sync,
    D::Item: Send + Sync,
    M: Fn(&D::Item, &D::Item) -> T + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    /// Parallel version of [`Self::new`].
    ///
    /// # Errors
    ///
    /// If `dataset` is empty.
    pub fn par_new<P, Ann>(mut dataset: D, metric: M, strategy: &PartitionStrategy<P>, annotator: &Ann) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
        Ann: Fn(&Cluster<T, A>) -> A + Send + Sync,
    {
        if dataset.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items in parallel", dataset.cardinality());

        let mut cluster_map = HashMap::new();

        // The `frontier` holds clusters that were just created but whose children have not yet been created.
        let mut frontier = vec![Cluster::par_new(dataset.as_mut_slice(), &metric, strategy)];
        ftlog::info!("Created root cluster with cardinality {}", frontier[0].0.cardinality);

        while let Some((mut cluster, splits)) = frontier.pop() {
            // For each split, create the child cluster and get the splits for its children and add them to the frontier.
            frontier.extend(splits.into_iter().map(|(offset, child_items)| {
                // Create child cluster and get items for its children.
                let (mut child, mut child_splits) = Cluster::par_new(child_items, &metric, strategy);

                ftlog::info!(
                    "Created a child cluster with cardinality {}, parent center index {}",
                    child.cardinality,
                    cluster.center_index
                );

                // Increment relevant indices
                child.depth = cluster.depth + 1;
                child.increment_indices(offset);
                for (ci, _) in &mut child_splits {
                    *ci += offset;
                }

                // Set parent center index for child cluster.
                child.parent_center_index = Some(cluster.center_index);
                (child, child_splits)
            }));

            // Annotate cluster and insert into map.
            cluster.annotation = annotator(&cluster);
            cluster_map.insert(cluster.center_index, cluster);
        }

        ftlog::info!("Finished creating tree with {} items", dataset.cardinality());
        Ok(Self { dataset, metric, cluster_map })
    }
}

impl<D, M, T, A, B> Tree<D, M, T, (A, B)> {
    /// De-compounds the annotations of the clusters in the tree and returns a new tree with the de-compounded annotations along with the other annotations.
    ///
    /// See [`Cluster::compound_annotation`] and [`Cluster::decompound_annotation`] for more details on how annotations are compounded and de-compounded.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///
    /// - A new `Tree` with the same items, the same metric, but with the annotations of the clusters de-compounded.
    /// - A `HashMap` mapping each cluster's center index to the de-compounded part of its annotation.
    pub fn decompound_annotations(self) -> (Tree<D, M, T, A>, HashMap<usize, B>) {
        let (dataset, metric, cluster_map) = self.into_parts();

        let (cluster_map, annotations_map) = cluster_map
            .into_iter()
            .map(|(ci, cluster)| {
                let (cluster, b) = cluster.decompound_annotation();
                ((ci, cluster), (ci, b))
            })
            .unzip();

        (Tree { dataset, metric, cluster_map }, annotations_map)
    }
}

/// Serialization and deserialization methods for [`Tree`], gated by the `serde` feature.
///
/// These methods will only serialize and deserialize the items and the cluster-map as a tuple. They will ignore the metric. This is because the metric is
/// typically a closure or function pointer, which cannot be serialized or deserialized. After deserialization, the metric must be provided using the
/// [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<D, T, A, M> Tree<D, M, T, A>
where
    D: serde::Serialize + serde::de::DeserializeOwned,
    T: serde::Serialize + serde::de::DeserializeOwned,
    A: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Serializes the `Tree` using Serde.
    ///
    /// # Errors
    ///
    /// If serialization fails.
    pub fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.dataset, &self.cluster_map).serialize(serializer)
    }

    /// Deserializes a `Tree` using Serde.
    ///
    /// # Errors
    ///
    /// If deserialization fails.
    pub fn deserialize<'de, De: serde::Deserializer<'de>>(deserializer: De, metric: M) -> Result<Self, De::Error> {
        let (dataset, cluster_map) = <(_, _)>::deserialize(deserializer)?;
        Ok(Self { dataset, metric, cluster_map })
    }
}

/// Implementation of [`databuf::Encode`] for [`Tree`], gated by the `serde` feature.
///
/// This does not serialize the metric. After deserialization, the metric must be provided using the [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<D, M, T, A> databuf::Encode for Tree<D, M, T, A>
where
    D: Dataset + databuf::Encode,
    T: databuf::Encode,
    A: databuf::Encode,
{
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        self.dataset.encode::<CONFIG>(buffer)?;
        self.cluster_map.encode::<CONFIG>(buffer)
    }
}

/// Implementation of [`databuf::Decode`] for [`Tree`], gated by the `serde` feature.
///
/// This sets a dummy metric during deserialization. After deserialization, the metric must be provided using the [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<'de, D, T, A> databuf::Decode<'de> for Tree<D, Box<dyn Fn(&D::Item, &D::Item) -> T>, T, A>
where
    D: Dataset + databuf::Decode<'de>,
    T: databuf::Decode<'de> + DistanceValue,
    A: databuf::Decode<'de>,
{
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let dataset = databuf::Decode::decode::<CONFIG>(buffer)?;
        let cluster_map = databuf::Decode::decode::<CONFIG>(buffer)?;
        let metric = Box::new(|_: &D::Item, _: &D::Item| T::zero()); // Placeholder; actual metric must be provided externally
        Ok(Self { dataset, metric, cluster_map })
    }
}
