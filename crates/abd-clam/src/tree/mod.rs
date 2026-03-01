//! A `Tree` of `Clusters` for use in CLAM.

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::DistanceValue;

mod cluster;
mod partition;

pub use cluster::Cluster;
pub use partition::strategy::{self as partition_strategy, PartitionStrategy};

/// The `Tree` struct is the main data structure used in CLAM.
///
/// If contains the root `Cluster`, the items stored in it, and the metric used to compute distances between items.
///
/// # Type Parameters
///
/// - `Id`: The type of the metadata associated with each item in the dataset.
/// - `I`: The type of the items in the dataset.
/// - `T`: The type of the distance values used in the tree.
/// - `A`: The type of any annotations that can be added to clusters.
/// - `M`: The type of the metric function used to compute distances between items from the dataset.
#[must_use]
#[derive(Clone, Debug)]
pub struct Tree<Id, I, T, A, M> {
    /// The items and their metadata used in the tree.
    pub(crate) items: Vec<(Id, I)>,
    /// All clusters in the tree. This is a mapping from `cluster.center_index` to `cluster`.
    pub(crate) cluster_map: HashMap<usize, Cluster<T, A>>,
    /// The metric used to compute distances between items.
    pub(crate) metric: M,
}

impl<I, T, M> Tree<usize, I, T, (), M>
where
    M: Fn(&I, &I) -> T,
    T: DistanceValue,
{
    /// Creates a new `Tree` from the given dataset and metric using the default partition strategy.
    ///
    /// This is a convenience method for using the original index of the items as their identifiers and for not using any annotations.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str> {
        let items = items.into_iter().enumerate().collect();
        Self::new(items, metric, &PartitionStrategy::default(), &|_| ())
    }

    /// Parallel version of [`Self::new_minimal`].
    ///
    /// # Errors
    ///
    /// See [`Self::new_minimal`].
    pub fn par_new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str>
    where
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
    {
        let items = items.into_iter().enumerate().collect();
        Self::par_new(items, metric, &PartitionStrategy::default(), &|_| ())
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M> {
    /// Provides ownership of the members of the `Tree`.
    #[expect(clippy::type_complexity)]
    pub fn into_parts(self) -> (Vec<(Id, I)>, HashMap<usize, Cluster<T, A>>, M) {
        (self.items, self.cluster_map, self.metric)
    }

    /// Creates a `Tree` from its members.
    pub(crate) const fn from_parts(items: Vec<(Id, I)>, cluster_map: HashMap<usize, Cluster<T, A>>, metric: M) -> Self {
        Self { items, cluster_map, metric }
    }

    /// Returns a reference to the dataset used in the tree.
    pub const fn items(&self) -> &Vec<(Id, I)> {
        &self.items
    }

    /// Returns a reference to the metric used in the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

    /// Returns a reference to the hash map of all clusters in the tree.
    pub const fn cluster_map(&self) -> &HashMap<usize, Cluster<T, A>> {
        &self.cluster_map
    }

    /// Returns the number of items in the tree.
    pub const fn cardinality(&self) -> usize {
        self.items.len()
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

    /// Returns all clusters in the tree in sorted order of their center indices, i.e. in pre-order traversal over the tree.
    pub fn sorted_clusters(&self) -> Vec<&Cluster<T, A>> {
        let mut clusters = self.cluster_map.values().collect::<Vec<_>>();
        clusters.sort_by_key(|c| c.center_index);
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
            .map(|center_indices| center_indices.iter().filter_map(|ci| self.cluster_map.get(ci)).collect())
    }
}

/// Various setters for `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M> {
    /// Changes the metric used in the tree to the provided one.
    pub fn with_metric<NewM>(self, metric: NewM) -> Tree<Id, I, T, A, NewM> {
        Tree {
            items: self.items,
            cluster_map: self.cluster_map,
            metric,
        }
    }
}

/// Constructors for `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    M: Fn(&I, &I) -> T,
    T: DistanceValue,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Arguments
    ///
    /// * `items` - A vec of items and their identifiers.
    /// * `metric` - A function that computes the distance between two items.
    /// * `strategy` - A `PartitionStrategy` that defines how to partition clusters.
    /// * `annotator` - A function that annotates clusters as they are created but before deciding whether they will be partitioned.
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new<P, Ann>(mut items: Vec<(Id, I)>, metric: M, strategy: &PartitionStrategy<P>, annotator: &Ann) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool,
        Ann: Fn(&Cluster<T, A>) -> A,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items", items.len());

        let mut cluster_map = HashMap::new();

        // The `frontier` holds clusters that were just created but whose children have not yet been created.
        let mut frontier = vec![Cluster::new(0, 0, None, &mut items, &metric, annotator, strategy)];
        while let Some((cluster, splits)) = frontier.pop() {
            // For each split, create the child cluster and get the splits for its children and add them to the frontier.
            frontier.extend(splits.into_iter().map(|(child_center_index, child_items)| {
                Cluster::new(
                    cluster.depth + 1,
                    child_center_index,
                    Some(cluster.center_index),
                    child_items,
                    &metric,
                    annotator,
                    strategy,
                )
            }));

            ftlog::info!(
                "Finished processing cluster with center index {}, depth {}, cardinality {} and child center indices {:?}",
                cluster.center_index,
                cluster.depth,
                cluster.cardinality,
                cluster.child_center_indices()
            );

            // Insert cluster into map.
            cluster_map.insert(cluster.center_index, cluster);
        }

        ftlog::info!("Finished creating tree with {} items", items.len());
        Ok(Self { items, cluster_map, metric })
    }
}

/// Parallelized constructors for `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    /// Parallel version of [`Self::new`].
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new<P, Ann>(mut items: Vec<(Id, I)>, metric: M, strategy: &PartitionStrategy<P>, annotator: &Ann) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
        Ann: Fn(&Cluster<T, A>) -> A + Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items in parallel", items.len());

        let mut cluster_map = HashMap::new();

        // The `frontier` holds clusters that were just created but whose children have not yet been created.
        let mut frontier = vec![Cluster::par_new(0, 0, None, &mut items, &metric, annotator, strategy)];
        while let Some((cluster, splits)) = frontier.pop() {
            // For each split, create the child cluster and get the splits for its children and add them to the frontier.
            frontier.extend(splits.into_iter().map(|(child_center_index, child_items)| {
                Cluster::par_new(
                    cluster.depth + 1,
                    child_center_index,
                    Some(cluster.center_index),
                    child_items,
                    &metric,
                    annotator,
                    strategy,
                )
            }));

            ftlog::info!(
                "Finished processing cluster with center index {}, depth {}, cardinality {} and child center indices {:?}",
                cluster.center_index,
                cluster.depth,
                cluster.cardinality,
                cluster.child_center_indices()
            );

            // Insert cluster into map.
            cluster_map.insert(cluster.center_index, cluster);
        }

        ftlog::info!("Finished creating tree with {} items", items.len());
        Ok(Self { items, cluster_map, metric })
    }
}

impl<Id, I, T, A, B, M> Tree<Id, I, T, (A, B), M> {
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
    #[expect(clippy::type_complexity)]
    pub fn decompound_annotations(self) -> (Tree<Id, I, T, A, M>, HashMap<usize, B>) {
        let (items, cluster_map, metric) = self.into_parts();

        let (cluster_map, annotations_map) = cluster_map
            .into_iter()
            .map(|(ci, cluster)| {
                let (cluster, b) = cluster.decompound_annotation();
                ((ci, cluster), (ci, b))
            })
            .unzip();

        (Tree { items, cluster_map, metric }, annotations_map)
    }
}

/// Serialization and deserialization methods for [`Tree`], gated by the `serde` feature.
///
/// These methods will only serialize and deserialize the items and the cluster-map as a tuple. They will ignore the metric. This is because the metric is
/// typically a closure or function pointer, which cannot be serialized or deserialized. After deserialization, the metric must be provided using the
/// [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: serde::Serialize + serde::de::DeserializeOwned,
    I: serde::Serialize + serde::de::DeserializeOwned,
    T: serde::Serialize + serde::de::DeserializeOwned,
    A: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Serializes the `Tree` using Serde.
    ///
    /// # Errors
    ///
    /// If serialization fails.
    pub fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.items, &self.cluster_map).serialize(serializer)
    }

    /// Deserializes a `Tree` using Serde.
    ///
    /// # Errors
    ///
    /// If deserialization fails.
    pub fn deserialize<'de, De: serde::Deserializer<'de>>(deserializer: De, metric: M) -> Result<Self, De::Error> {
        let (items, cluster_map) = <(_, _)>::deserialize(deserializer)?;
        Ok(Self { items, cluster_map, metric })
    }
}

/// Implementation of [`databuf::Encode`] for [`Tree`], gated by the `serde` feature.
///
/// This does not serialize the metric. After deserialization, the metric must be provided using the [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<Id, I, T, A, M> databuf::Encode for Tree<Id, I, T, A, M>
where
    Id: databuf::Encode,
    I: databuf::Encode,
    T: databuf::Encode,
    A: databuf::Encode,
{
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        self.items.encode::<CONFIG>(buffer)?;
        self.cluster_map.encode::<CONFIG>(buffer)
    }
}

/// Implementation of [`databuf::Decode`] for [`Tree`], gated by the `serde` feature.
///
/// This sets a dummy metric during deserialization. After deserialization, the metric must be provided using the [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<'de, Id, I, T, A> databuf::Decode<'de> for Tree<Id, I, T, A, Box<dyn Fn(&I, &I) -> T>>
where
    Id: databuf::Decode<'de>,
    I: databuf::Decode<'de>,
    T: databuf::Decode<'de> + DistanceValue,
    A: databuf::Decode<'de>,
{
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let items = databuf::Decode::decode::<CONFIG>(buffer)?;
        let cluster_map = databuf::Decode::decode::<CONFIG>(buffer)?;
        let metric = Box::new(|_: &I, _: &I| T::zero()); // Placeholder; actual metric must be provided externally
        Ok(Self { items, cluster_map, metric })
    }
}
