//! A [`Tree`] stores a dataset of items in a hierarchical structure of [`Cluster`]s.

use tqdm;

use crate::DistanceValue;

mod cluster;
mod partition;

pub use cluster::Cluster;
pub use partition::strategy::{self as partition_strategy, PartitionStrategy};

/// A `Tree` stores a dataset of items in a hierarchical structure of [`Cluster`]s.
///
/// # Type Parameters
///
/// - `Id`: The type of the metadata associated with each item, `I`, in the dataset.
/// - `I`: The type of the items in the dataset.
/// - `T`: The type of the distance values used in the tree.
/// - `A`: The type of annotations that can be added to clusters.
/// - `M`: The type of the metric function used to compute distances between items from the dataset.
///
/// In order to build a `Tree`, one must provide:
///
/// - A dataset of items, along with the metadata for each item, in the form of a `Vec<(Id, I)>`.
/// - A metric function that can compute distances between items in the dataset, of type `M: Fn(&I, &I) -> T`.
/// - An annotator function that can annotate clusters as they are created but before deciding whether they will be partitioned.
/// - A predicate function that decides whether to partition a cluster.
/// - A [`PartitionStrategy`] that defines how to partition clusters after deciding that they should be partitioned.
///
/// After the tree is built, the items (and their associated metadata) will have been reordered to be in a depth-first traversal order of the [`Cluster`]s in
/// the tree. Thus, each [`Cluster`] represents a contiguous subsequence of the items in the `items` vector in the `Tree`, and the indices of that subsequence
/// are called the "item indices" of the [`Cluster`].
///
/// The "root" [`Cluster`] of the tree represents all items in the dataset, and the center of this [`Cluster`] is the item at index `0` in the `items` vector.
///
/// See [`Cluster`] for more details on the structure of clusters and how they relate to the items in the tree.
#[must_use]
#[derive(Clone, Debug)]
pub struct Tree<Id, I, T, A, M> {
    /// The items, their metadata, and cluster locations in the tree.
    pub(crate) items: Vec<(Id, I, ClusterLocation<T, A>)>,
    /// The metric used to compute distances between items.
    pub(crate) metric: M,
}

/// The implementation of [`deepsize::DeepSizeOf`] for [`Tree`] ignores the metric, because the metric is typically a function pointer.
#[cfg(feature = "pancakes")]
impl<Id, I, T, A, M> deepsize::DeepSizeOf for Tree<Id, I, T, A, M>
where
    Id: deepsize::DeepSizeOf,
    I: deepsize::DeepSizeOf,
    T: deepsize::DeepSizeOf,
    A: deepsize::DeepSizeOf,
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.items.deep_size_of_children(context)
    }
}

/// A helper enum for storing clusters in the tree.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub(crate) enum ClusterLocation<T, A> {
    /// A cluster
    Cluster(Cluster<T, A>),
    /// An index of the cluster responsible for the given item.
    CenterIndex(usize),
}

impl<T, A> From<Cluster<T, A>> for ClusterLocation<T, A> {
    fn from(cluster: Cluster<T, A>) -> Self {
        Self::Cluster(cluster)
    }
}

impl<T, A> From<usize> for ClusterLocation<T, A> {
    fn from(ci: usize) -> Self {
        Self::CenterIndex(ci)
    }
}

impl<T, A> From<ClusterLocation<T, A>> for Option<Cluster<T, A>> {
    fn from(loc: ClusterLocation<T, A>) -> Self {
        match loc {
            ClusterLocation::Cluster(cluster) => Some(cluster),
            ClusterLocation::CenterIndex(_) => None,
        }
    }
}

impl<T, A> ClusterLocation<T, A> {
    /// Returns a reference to the cluster if this is a `ClusterLocation::Cluster`, and `None` otherwise.
    pub(crate) const fn as_cluster(&self) -> Option<&Cluster<T, A>> {
        match self {
            Self::Cluster(cluster) => Some(cluster),
            Self::CenterIndex(_) => None,
        }
    }

    /// Returns a mutable reference to the cluster if this is a `ClusterLocation::Cluster`, and `None` otherwise.
    pub(crate) const fn as_cluster_mut(&mut self) -> Option<&mut Cluster<T, A>> {
        match self {
            Self::Cluster(cluster) => Some(cluster),
            Self::CenterIndex(_) => None,
        }
    }
}

impl<I, T, M> Tree<usize, I, T, (), M>
where
    M: Fn(&I, &I) -> T,
    T: DistanceValue,
{
    /// Creates a new `Tree` from the given dataset and metric.
    ///
    /// This is a convenience method for using the original index of the items as their identifiers, for not using any annotations, always partitioning
    /// clusters with cardinality greater than 2, and using the default partition strategy.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str> {
        let items = items.into_iter().enumerate().collect();
        Self::new(items, metric, &|_| (), &|c: &_| c.cardinality > 2, &PartitionStrategy::default())
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
        Self::par_new(items, metric, &|_| (), &|c: &_| c.cardinality > 2, &PartitionStrategy::default())
    }

    /// Creates a new binary `Tree` from the given dataset and metric.
    ///
    /// This is a convenience method for using the original index of the items as their identifiers, for not using any annotations, always partitioning clusters
    /// with cardinality greater than 2.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn new_binary(items: Vec<I>, metric: M) -> Result<Self, &'static str> {
        let items = items.into_iter().enumerate().collect();
        Self::new(items, metric, &|_| (), &|c: &_| c.cardinality > 2, &PartitionStrategy::binary())
    }

    /// Parallel version of [`Self::new_binary`].
    ///
    /// # Errors
    ///
    /// See [`Self::new_binary`].
    pub fn par_new_binary(items: Vec<I>, metric: M) -> Result<Self, &'static str>
    where
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
    {
        let items = items.into_iter().enumerate().collect();
        Self::par_new(items, metric, &|_| (), &|c: &_| c.cardinality > 2, &PartitionStrategy::binary())
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M> {
    /// Returns a reference to the metric used in the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

    /// Swaps the metric used in the tree with the given metric and returns a new tree with the new metric and the same items.
    ///
    /// This is useful for deserialization, where the metric is not serialized and must be provided after deserialization.
    pub fn with_metric<NewM>(self, metric: NewM) -> Tree<Id, I, T, A, NewM> {
        let Self { items, .. } = self;
        Tree { items, metric }
    }

    /// Returns the number of items in the tree.
    pub const fn cardinality(&self) -> usize {
        self.items.len()
    }

    /// Returns a reference to the cluster in the tree that directly contains the item with the given index.
    ///
    /// If the indexed item is the center of a cluster, then that cluster is returned. Otherwise, the item must a non-center in a leaf cluster, and that leaf
    /// cluster is returned.
    ///
    /// If the index is out of bounds, this returns `None`.
    pub fn get_cluster(&self, index: usize) -> Option<&Cluster<T, A>> {
        let index = match self.items.get(index)?.2 {
            ClusterLocation::Cluster(_) => index,
            ClusterLocation::CenterIndex(center_index) => center_index,
        };
        match &self.items[index].2 {
            ClusterLocation::Cluster(cluster) => Some(cluster),
            ClusterLocation::CenterIndex(_) => unreachable!("Location always indexes to a Cluster {}", index),
        }
    }

    /// The same as [`Self::get_cluster`], but does not check whether the index is out of bounds or if the indexed item is not a cluster center.
    pub(crate) fn get_cluster_unchecked(&self, index: usize) -> &Cluster<T, A> {
        match &self.items[index].2 {
            ClusterLocation::Cluster(cluster) => cluster,
            ClusterLocation::CenterIndex(_) => unreachable!("Location always indexes to a Cluster {}", index),
        }
    }

    /// The same as [`Self::get_cluster_unchecked`], but returns a mutable reference to the cluster.
    pub(crate) fn get_cluster_unchecked_mut(&mut self, index: usize) -> &mut Cluster<T, A> {
        match &mut self.items[index].2 {
            ClusterLocation::Cluster(cluster) => cluster,
            ClusterLocation::CenterIndex(_) => unreachable!("Location always indexes to a Cluster {}", index),
        }
    }

    /// The same as [`Self::get_cluster`], but returns a mutable reference to the cluster.
    pub fn get_cluster_mut(&mut self, index: usize) -> Option<&mut Cluster<T, A>> {
        let index = match self.items.get(index)?.2 {
            ClusterLocation::Cluster(_) => index,
            ClusterLocation::CenterIndex(center_index) => center_index,
        };
        match &mut self.items[index].2 {
            ClusterLocation::Cluster(cluster) => Some(cluster),
            ClusterLocation::CenterIndex(_) => unreachable!("Location always indexes to a Cluster {}", index),
        }
    }

    /// Returns a reference to the (Id, Item) pair for the item with the given index, None if the index is out of bounds.
    pub fn get_item(&self, index: usize) -> Option<(&Id, &I)> {
        self.items.get(index).map(|(id, item, _)| (id, item))
    }

    /// Returns an iterator over the items, their identifiers, and the clusters they are the center of, if any.
    pub fn iter_items(&self) -> impl Iterator<Item = (&Id, &I, Option<&Cluster<T, A>>)> {
        self.items.iter().map(|(id, item, loc)| (id, item, loc.as_cluster()))
    }

    /// Returns an iterator over all clusters in the tree.
    pub fn iter_clusters(&self) -> impl Iterator<Item = &Cluster<T, A>> {
        self.items.iter().filter_map(|(_, _, loc)| loc.as_cluster())
    }

    /// Provides ownership of the items, clusters, and metric in the tree. This consumes the tree.
    #[expect(clippy::type_complexity)]
    pub fn take_members(self) -> (Vec<(Id, I)>, Vec<Cluster<T, A>>, M) {
        let Self { items, metric } = self;
        let (items, locations): (Vec<_>, Vec<_>) = items.into_iter().map(|(id, item, loc)| ((id, item), loc)).unzip();
        let clusters = locations.into_iter().filter_map(Into::into).collect();
        (items, clusters, metric)
    }

    /// Returns the number of clusters in the tree.
    pub fn n_clusters(&self) -> usize {
        self.iter_clusters().count()
    }

    /// Returns a reference to the root cluster of the tree.
    pub fn root(&self) -> &Cluster<T, A> {
        self.items[0]
            .2
            .as_cluster()
            .unwrap_or_else(|| unreachable!("Root cluster should be at index 0"))
    }

    /// Returns references to the children of the given cluster, if any.
    pub fn children_of(&self, cluster: &Cluster<T, A>) -> Option<Vec<&Cluster<T, A>>> {
        cluster
            .child_center_indices()
            .map(|center_indices| center_indices.iter().map(|&ci| self.get_cluster_unchecked(ci)).collect())
    }

    /// Returns the center-indices of the clusters that must be traversed to get from the root to arrive at the indexed item.
    ///
    /// This excludes the center index of the root cluster, which is always `0`, includes the index of the indexed item, and is sorted in ascending order, i.e.
    /// from parent to child.
    ///
    /// If the index is out of bounds, this returns `None`.
    pub fn path_to_item(&self, index: usize) -> Option<Vec<usize>> {
        if index >= self.cardinality() {
            return None;
        }
        Some(self.path_to_item_unchecked(index))
    }

    /// The same as [`Self::path_to_item`], but does not check whether the index is out of bounds, and panics if it is.
    pub(crate) fn path_to_item_unchecked(&self, index: usize) -> Vec<usize> {
        let mut path = Vec::new();

        let cid = match self.items[index].2 {
            ClusterLocation::Cluster(_) => index,
            ClusterLocation::CenterIndex(center_index) => {
                path.push(index);
                center_index
            }
        };

        let mut current_cluster = self.get_cluster_unchecked(cid);
        while let Some(parent_ci) = current_cluster.parent_center_index {
            path.push(current_cluster.center_index);
            current_cluster = self.get_cluster_unchecked(parent_ci);
        }
        path.reverse();

        path
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
    /// * `annotator` - A function that annotates clusters as they are created but before deciding whether they will be partitioned.
    /// * `should_partition` - A function that takes a `Cluster` and returns a boolean indicating whether to partition the cluster further. This is called after
    ///   `annotator` is called and can use the annotations added by `annotator` to make its decision.
    /// * `strategy` - A `PartitionStrategy` that defines how to partition clusters.
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new<Ann, P>(mut items: Vec<(Id, I)>, metric: M, annotator: &Ann, should_partition: &P, strategy: &PartitionStrategy) -> Result<Self, &'static str>
    where
        Ann: Fn(&Cluster<T, ()>) -> A,
        P: Fn(&Cluster<T, A>) -> bool,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items", items.len());

        let mut locations = {
            let mut locations = Vec::with_capacity(items.len());
            for _ in 0..items.len() {
                locations.push(ClusterLocation::CenterIndex(0));
            }
            locations
        };

        let mut progress_bar = tqdm::pbar(Some(items.len()));
        progress_bar.set_desc(Some(format!("Building tree with {} items", items.len())));

        // The `frontier` holds clusters that were just created but whose children have not yet been created.
        let mut frontier = vec![Cluster::new(0, 0, None, &mut items, &metric, annotator, should_partition, strategy)];
        progress_bar.update(1).map_err(|_| "Progress bar error")?;
        while let Some((cluster, splits)) = frontier.pop() {
            // For each split, create the child cluster and get the splits for its children and add them to the frontier.
            frontier.extend(splits.into_iter().rev().map(|(child_center_index, child_items)| {
                let c = Cluster::new(
                    cluster.depth + 1,
                    child_center_index,
                    Some(cluster.center_index),
                    child_items,
                    &metric,
                    annotator,
                    should_partition,
                    strategy,
                );
                progress_bar.update(1).unwrap_or_else(|e| unreachable!("Progress bar error: {e}"));
                c
            }));

            if let Some(cids) = cluster.child_center_indices() {
                ftlog::info!(
                    "Finished processing cluster with center index {}, depth {}, cardinality {} and child center indices {:?}",
                    cluster.center_index,
                    cluster.depth,
                    cluster.cardinality,
                    cids
                );
            } else {
                ftlog::info!(
                    "Finished processing leaf cluster with center index {}, depth {}, cardinality {}",
                    cluster.center_index,
                    cluster.depth,
                    cluster.cardinality,
                );
            }

            // Insert cluster into locations vector.
            let i = cluster.center_index;
            if cluster.is_leaf() && cluster.cardinality > 1 {
                for j in cluster.subtree_range() {
                    locations[j] = ClusterLocation::CenterIndex(i);
                }

                progress_bar.update(cluster.cardinality - 1).map_err(|_| "Progress bar error")?;
            }
            locations[i] = ClusterLocation::Cluster(cluster);
        }

        ftlog::info!("Finished creating tree with {} items", items.len());
        let items = items.into_iter().zip(locations).map(|((id, item), loc)| (id, item, loc)).collect();
        Ok(Self { items, metric })
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
    pub fn par_new<Ann, P>(
        mut items: Vec<(Id, I)>,
        metric: M,
        annotator: &Ann,
        should_partition: &P,
        strategy: &PartitionStrategy,
    ) -> Result<Self, &'static str>
    where
        Ann: Fn(&Cluster<T, ()>) -> A + Send + Sync,
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
    {
        profi::prof!("Tree::par_new");

        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items in parallel", items.len());

        let mut locations = {
            let mut locations = Vec::with_capacity(items.len());
            for _ in 0..items.len() {
                locations.push(ClusterLocation::CenterIndex(0));
            }
            locations
        };

        let mut progress_bar = tqdm::pbar(Some(items.len()));
        progress_bar.set_desc(Some(format!("Building tree, in parallel, with {} items", items.len())));

        // The `frontier` holds clusters that were just created but whose children have not yet been created.
        let mut frontier = vec![Cluster::par_new(0, 0, None, &mut items, &metric, annotator, should_partition, strategy)];
        progress_bar.update(1).map_err(|_| "Progress bar error")?;
        while let Some((cluster, splits)) = frontier.pop() {
            profi::prof!("Tree::par_new::frontier_pop");

            // For each split, create the child cluster and get the splits for its children and add them to the frontier.
            frontier.extend(splits.into_iter().rev().map(|(child_center_index, child_items)| {
                profi::prof!("Tree::par_new::frontier_extension");
                let c = Cluster::par_new(
                    cluster.depth + 1,
                    child_center_index,
                    Some(cluster.center_index),
                    child_items,
                    &metric,
                    annotator,
                    should_partition,
                    strategy,
                );
                progress_bar.update(1).unwrap_or_else(|e| unreachable!("Progress bar error: {e}"));
                c
            }));

            if let Some(cids) = cluster.child_center_indices() {
                ftlog::info!(
                    "Finished processing cluster with center index {}, depth {}, cardinality {} and child center indices {:?}",
                    cluster.center_index,
                    cluster.depth,
                    cluster.cardinality,
                    cids
                );
            } else {
                ftlog::info!(
                    "Finished processing leaf cluster with center index {}, depth {}, cardinality {}",
                    cluster.center_index,
                    cluster.depth,
                    cluster.cardinality,
                );
            }

            // Insert cluster into locations vector.
            let i = cluster.center_index;
            if cluster.is_leaf() && cluster.cardinality > 1 {
                profi::prof!("Tree::par_new::locations_update");
                for j in cluster.subtree_range() {
                    locations[j] = ClusterLocation::CenterIndex(i);
                }
                progress_bar.update(cluster.cardinality - 1).map_err(|_| "Progress bar error")?;
            }
            locations[i] = ClusterLocation::Cluster(cluster);
        }

        ftlog::info!("Finished creating tree with {} items", items.len());
        let items = items.into_iter().zip(locations).map(|((id, item), loc)| (id, item, loc)).collect();
        Ok(Self { items, metric })
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
    /// - A `Vec` of the de-compounded annotations for each cluster in the tree, where the annotation at index `i` is `Some` if there is a cluster whose center
    ///   index is `i`, and `None` otherwise.
    #[expect(clippy::type_complexity)]
    pub fn decompound_annotations(self) -> (Tree<Id, I, T, A, M>, Vec<Option<B>>) {
        let Self { items, metric } = self;

        let (items, annotations) = items
            .into_iter()
            .map(|(id, item, loc)| {
                let (loc, ann) = match loc {
                    ClusterLocation::Cluster(cluster) => {
                        let (cluster, b) = cluster.decompound_annotation();
                        (ClusterLocation::Cluster(cluster), Some(b))
                    }
                    ClusterLocation::CenterIndex(ci) => (ClusterLocation::CenterIndex(ci), None),
                };
                ((id, item, loc), ann)
            })
            .unzip();

        (Tree { items, metric }, annotations)
    }
}

/// Implementation of [`serde::Serialize`] for [`Tree`].
///
/// This does not serialize the metric. After deserialization, the metric must be provided using the [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<Id, I, T, A, M> serde::Serialize for Tree<Id, I, T, A, M>
where
    Id: serde::Serialize,
    I: serde::Serialize,
    T: serde::Serialize,
    A: serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.items.serialize(serializer)
    }
}

/// Implementation of [`serde::Deserialize`] for [`Tree`].
///
/// Since the metric is never serialized, the deserialized tree has a unit type `()` for its metric. After deserialization, the metric must be provided using
/// the [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<'de, Id, I, T, A> serde::Deserialize<'de> for Tree<Id, I, T, A, ()>
where
    Id: serde::de::Deserialize<'de>,
    I: serde::de::Deserialize<'de>,
    T: serde::de::Deserialize<'de>,
    A: serde::de::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let items = <_>::deserialize(deserializer)?;
        Ok(Self { items, metric: () })
    }
}

/// Implementation of [`databuf::Encode`] for [`Tree`].
///
/// This does not encode the metric. After decoding, the metric must be provided using the [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<Id, I, T, A, M> databuf::Encode for Tree<Id, I, T, A, M>
where
    Id: databuf::Encode,
    I: databuf::Encode,
    T: databuf::Encode,
    A: databuf::Encode,
{
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        self.items.encode::<CONFIG>(buffer)
    }
}

/// Implementation of [`databuf::Decode`] for [`Tree`].
///
/// Since the metric is never encoded, the decoded tree has a unit type `()` for its metric. After decoding, the metric must be provided using the
/// [`Tree::with_metric`] method.
#[cfg(feature = "serde")]
impl<'de, Id, I, T, A> databuf::Decode<'de> for Tree<Id, I, T, A, ()>
where
    Id: databuf::Decode<'de>,
    I: databuf::Decode<'de>,
    T: databuf::Decode<'de>,
    A: databuf::Decode<'de>,
{
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let items = databuf::Decode::decode::<CONFIG>(buffer)?;
        Ok(Self { items, metric: () })
    }
}
