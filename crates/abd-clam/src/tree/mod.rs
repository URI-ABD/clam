//! A `Tree` of `Clusters` for use in CLAM.

use std::collections::HashMap;

use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::DistanceValue;

mod cluster;
mod partition;

pub use cluster::Cluster;
pub use partition::strategy::{self as partition_strategy, PartitionStrategy};

// TODO(Najib): Add methods to annotate clusters after tree creation.

/// The `Tree` struct is the main data structure used in CLAM.
///
/// If contains the root `Cluster`, the items stored in it, and the metric used to compute distances between items.
///
/// # Type Parameters
///
/// - `Id`: The type of the identifier for each item in the tree.
/// - `I`: The type of the items stored in the tree.
/// - `T`: The type of the distance values used in the tree.
/// - `A`: The type of any annotations that can be added to clusters.
/// - `M`: The type of the metric function used to compute distances between items.
#[must_use]
#[derive(Clone, Debug)]
pub struct Tree<Id, I, T, A, M> {
    /// The items stored in the tree, each paired with its identifier.
    pub(crate) items: Vec<(Id, I)>,
    /// All clusters in the tree. This is a mapping from `cluster.center_index` to `cluster`.
    pub(crate) cluster_map: HashMap<usize, Cluster<T, A>>,
    /// The metric used to compute distances between items.
    pub(crate) metric: M,
}

/// Minimal constructors for `Tree`.
///
/// - The identifier type is set to `usize` and will be the index of the item in the original vector.
/// - The annotation type is set to `()`, meaning that no annotations are stored in the tree.
/// - The default [`PartitionStrategy`](PartitionStrategy) is used to build a binary tree.
impl<I, T, M> Tree<usize, I, T, (), M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str> {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let items = items.into_iter().enumerate().collect::<Vec<_>>();
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
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let items = items.into_iter().enumerate().collect::<Vec<_>>();
        Self::par_new(items, metric, &PartitionStrategy::default(), &|_| ())
    }
}

/// Various getter methods for `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M> {
    /// Provides ownership of the members of the `Tree`.
    #[expect(clippy::type_complexity)]
    pub fn into_parts(self) -> (Vec<(Id, I)>, HashMap<usize, Cluster<T, A>>, M) {
        (self.items, self.cluster_map, self.metric)
    }

    /// Returns a reference to all items in the tree.
    pub fn items(&self) -> &[(Id, I)] {
        &self.items
    }

    /// Consumes the tree and returns all items stored in it.
    pub fn take_items(self) -> Vec<(Id, I)> {
        self.items
    }

    /// Returns the number of items stored in the tree.
    pub const fn cardinality(&self) -> usize {
        self.items.len()
    }

    /// Returns a reference to the hash map of all clusters in the tree.
    pub const fn cluster_map(&self) -> &HashMap<usize, Cluster<T, A>> {
        &self.cluster_map
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

    /// Returns a reference to the metric used in the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

    /// Changes the metric used in the tree to the provided one.
    pub fn with_metric<N>(self, metric: N) -> Tree<Id, I, T, A, N> {
        Tree {
            items: self.items,
            cluster_map: self.cluster_map,
            metric,
        }
    }

    /// Returns a reference to the root cluster of the tree.
    pub fn root(&self) -> &Cluster<T, A> {
        self.cluster_map
            .get(&0)
            .unwrap_or_else(|| unreachable!("Tree must have a root cluster with center_index 0"))
    }

    /// Returns references to the children of the given cluster, if any.
    pub fn children_of(&self, cluster: &Cluster<T, A>) -> Option<Vec<&Cluster<T, A>>> {
        cluster.child_center_indices().map(|center_indices| {
            center_indices
                .iter()
                .map(|&ci| {
                    self.cluster_map
                        .get(&ci)
                        .unwrap_or_else(|| unreachable!("Invalid center index {ci} from parent {}", cluster.center_index()))
                })
                .collect()
        })
    }
}

/// Constructors and methods for computing distances in `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Arguments
    ///
    /// * `items` - A vector of tuples, each containing an identifier and an item.
    /// * `metric` - A function that computes the distance between two items.
    /// * `strategy` - A `PartitionStrategy` that defines how to partition clusters.
    /// * `annotator` - A function that annotates clusters as they are created.
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
        let mut frontier = vec![Cluster::new(&mut items, &metric, strategy)];
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
                child.depth += 1;
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

        ftlog::info!("Finished creating tree with {} items", items.len());
        Ok(Self { items, cluster_map, metric })
    }

    /// Returns the distance between the query item and the item with the given index.
    pub fn distance_to(&self, query: &I, i: usize) -> T {
        (self.metric)(query, &self.items[i].1)
    }

    /// Returns the distance between the query item and the center of the given cluster.
    pub fn distance_to_center(&self, query: &I, cluster: &Cluster<T, A>) -> T {
        self.distance_to(query, cluster.center_index())
    }

    /// Returns the distances between the query item and all items in the given cluster, excluding the cluster's center.
    pub fn distances_to_items_in_subtree(&self, query: &I, cluster: &Cluster<T, A>) -> impl Iterator<Item = (usize, T)> {
        cluster.items_indices().skip(1).map(|i| (i, self.distance_to(query, i)))
    }

    /// Returns the distances between the query item and all items in the given cluster, including the cluster's center.
    pub fn distances_to_items_in_cluster(&self, query: &I, cluster: &Cluster<T, A>) -> impl Iterator<Item = (usize, T)> {
        cluster.items_indices().map(|i| (i, self.distance_to(query, i)))
    }
}

/// Parallelized constructors and methods for computing distances in `Tree`.
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
        let mut frontier = vec![Cluster::par_new(&mut items, &metric, strategy)];
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
                child.depth += 1;
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

        ftlog::info!("Finished creating tree with {} items", items.len());
        Ok(Self { items, cluster_map, metric })
    }

    /// Parallel version of [`distances_to_items_in_subtree`](Self::distances_to_items_in_subtree).
    pub fn par_distances_to_items_in_subtree(&self, query: &I, cluster: &Cluster<T, A>) -> impl ParallelIterator<Item = (usize, T)> {
        cluster.items_indices().into_par_iter().skip(1).map(|i| (i, self.distance_to(query, i)))
    }

    /// Parallel version of [`distances_to_items_in_cluster`](Self::distances_to_items_in_cluster).
    pub fn par_distances_to_items_in_cluster(&self, query: &I, cluster: &Cluster<T, A>) -> impl ParallelIterator<Item = (usize, T)> {
        cluster.items_indices().into_par_iter().map(|i| (i, self.distance_to(query, i)))
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
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D, metric: M) -> Result<Self, D::Error> {
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
