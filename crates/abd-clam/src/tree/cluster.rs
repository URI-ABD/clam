//! A `Cluster` in a `Tree` for use in CLAM.

use crate::DistanceValue;

/// A `Cluster` in a [`Tree`](crate::Tree) represents a collection of "similar" items in the tree.
///
/// # Type Parameters
///
/// - `T`: The type of the distance values between items.
/// - `A`: The type of the annotation data associated with this cluster.
///
/// The number of items represented by a `Cluster` is given by its [`cardinality`](Self::cardinality), and the items represented by the cluster are located in
/// the `center_index..(center_index + cardinality)` range of the `items` vector of the `Tree`. This range can be retrieved using the
/// [`items_range`](Self::items_range) method of the `Cluster`.
///
/// Every `Cluster` has a "center" item, which is the [geometric median](https://en.wikipedia.org/wiki/Geometric_median) of the items (or a smaller sample of
/// the items) represented by the cluster. This item is located at [`center_index`](Self::center_index) in the `items` array of the `Tree`.
///
/// The [`radius`](Self::radius) of a `Cluster` is the distance from the center item to the furthest item represented by the cluster. Note that a `Cluster` may
/// not necessarily represent all items within its `radius`.
///
/// The `Cluster` may also have "child clusters" if it was partitioned during the construction of the tree. If the cluster was partitioned then it is considered
/// a "parent cluster", and the indices of the centers of the child clusters can be retrieved using the [`child_center_indices`](Self::child_center_indices)
/// method. See [`PartitionStrategy`](crate::tree::PartitionStrategy) for more information on how clusters are partitioned.
///
/// The `span` of a cluster is the distance between two highly dissimilar items (though not necessarily the *most dissimilar pair* of items) in the cluster.
/// These two items were used to guide the partitioning of the cluster. The span can be retrieved using the [`span`](Self::span) method.
///
/// For any child cluster, the index of the center of its parent cluster can be retrieved using the [`parent_center_index`](Self::parent_center_index) method.
/// For the root cluster, this will be `None`.
///
/// The root cluster of the tree will have a [`depth`](Self::depth) of `0`, and the depth of each child cluster will be one greater than that of its parent.
///
/// While building a tree, once a `Cluster` has been created and we decide that it should be partitioned, the "center item" of the cluster is not given to its
/// children. As such, the index of the [`center_index`](Self::center_index) of a cluster uniquely identifies the cluster in the tree, and can be used to
/// retrieve the cluster from the tree using the [`Tree::get_cluster`](crate::Tree::get_cluster) method.
///
/// While building a tree, the `Cluster`s are annotated with additional information (of type `A`). These annotations can, later, be replaced or changed. These
/// annotations can be used to store additional information relevant to each cluster depending on the use-case. For example, annotations can be used to help
/// decide whether to partition the cluster when building the tree, to store the recursive and unitary compression costs of a cluster in the
/// [`compress`](crate::tree::Tree::compress) method from the [`pancakes`](crate::pancakes) module, to store guidelines for sequence alignment in the
/// [`musals`](crate::musals) module.
///
/// In summary, the relevant information about a `Cluster` includes:
///
/// - `depth`: The depth of the cluster in the tree, with the root cluster at depth `0`.
/// - `center_index`: The index of the center of this cluster among the `items` in the `Tree`.
/// - `cardinality`: The number of items in the subtree rooted at this cluster, including the center item.
/// - `radius`: The distance from the center item to the furthest item represented by the cluster.
/// - `lfd`: The Local Fractal Dimension of the cluster.
/// - `children`: If the cluster was partitioned, the indices of the centers of the child cluster centers and the span of this cluster.
/// - `annotation`: Arbitrary data associated with this cluster.
/// - `parent_center_index`: The index of the center of the parent cluster. For the root cluster, this will be `None`.
#[must_use]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub struct Cluster<T, A> {
    /// Depth of this cluster in the tree, with root at depth 0.
    pub(crate) depth: usize,
    /// Index of the center item in the `items` array of the `Tree`.
    pub(crate) center_index: usize,
    /// Number of items in the subtree rooted at this cluster, including the center item.
    pub(crate) cardinality: usize,
    /// The distance from the center item to the furthest item in the cluster.
    pub(crate) radius: T,
    /// The Local Fractal Dimension of the cluster.
    pub(crate) lfd: f64,
    /// The indices of child centers and the span of this cluster, if it was partitioned. The span is the distance between the two poles used to partition the cluster.
    pub(crate) children: Option<(Box<[usize]>, T)>,
    /// Arbitrary data associated with this cluster.
    pub(crate) annotation: A,
    /// The index of the center of the parent cluster. For the root cluster, this will be None.
    pub(crate) parent_center_index: Option<usize>,
}

impl<T, A> PartialEq for Cluster<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.center_index == other.center_index
    }
}

impl<T, A> Eq for Cluster<T, A> {}

impl<T, A> PartialOrd for Cluster<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, A> Ord for Cluster<T, A> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.center_index.cmp(&other.center_index)
    }
}

impl<T: DistanceValue, A> Cluster<T, A> {
    /// Gets the radius of this cluster.
    pub const fn radius(&self) -> T {
        self.radius
    }

    /// If the cluster was partitioned, gets the indices of the centers of the children and the span of this cluster.
    pub fn child_center_indices_and_span(&self) -> Option<(&[usize], T)> {
        self.children
            .as_ref()
            .map(|(child_center_indices, span)| (child_center_indices.as_ref(), *span))
    }

    /// If the cluster was partitioned, gets the span of this cluster.
    pub fn span(&self) -> Option<T> {
        self.children.as_ref().map(|(_, span)| *span)
    }

    /// Returns true if this cluster is a singleton (i.e., contains exactly one item or has a radius of zero).
    pub fn is_singleton(&self) -> bool {
        self.cardinality == 1 || self.radius.is_zero()
    }
}

impl<T, A> Cluster<T, A> {
    /// Returns the depth of this cluster in the tree.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Gets the center item index.
    pub const fn center_index(&self) -> usize {
        self.center_index
    }

    /// Gets the cardinality of this cluster.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Gets the local fractal dimension of this cluster.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// If the cluster was partitioned, gets the center indices of the children of this cluster.
    pub fn child_center_indices(&self) -> Option<&[usize]> {
        self.children.as_ref().map(|(child_center_indices, _)| child_center_indices.as_ref())
    }

    /// Returns the index of the center of the parent cluster. For the root cluster, this will be `None`.
    pub const fn parent_center_index(&self) -> Option<usize> {
        self.parent_center_index
    }

    /// Returns a reference to the cluster's annotation.
    pub const fn annotation(&self) -> &A {
        &self.annotation
    }

    /// Returns a mutable reference to the cluster's annotation.
    pub const fn annotation_mut(&mut self) -> &mut A {
        &mut self.annotation
    }

    /// Returns true if this cluster is a leaf (i.e., has no children or was not partitioned).
    pub const fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Returns a [`Range`](core::ops::Range) of items represented by this cluster, including the center item.
    ///
    /// This can be used to index into the [`Tree`](crate::Tree) using the [`Tree::get_item`](crate::Tree::get_item) and
    /// [`Tree::get_cluster`](crate::Tree::get_cluster) methods, as well as their mutable counterparts.
    pub const fn items_range(&self) -> std::ops::Range<usize> {
        self.center_index..(self.center_index + self.cardinality)
    }

    /// The same as [`items_range`](Self::items_range), but excludes the center item.
    pub const fn subtree_range(&self) -> std::ops::Range<usize> {
        (self.center_index + 1)..(self.center_index + self.cardinality)
    }

    /// Replaces the annotation of this cluster, returning the old annotation.
    pub const fn replace_annotation(&mut self, annotation: A) -> A {
        core::mem::replace(&mut self.annotation, annotation)
    }

    /// Replaces the annotation of this cluster using the provided function, returning the old annotation.
    ///
    /// The provided function is called before the annotation is updated, so it can use the current annotation if needed.
    pub fn replace_annotation_with<F: Fn(&Self) -> A>(&mut self, f: &F) -> A {
        let new_annotation = f(self);
        self.replace_annotation(new_annotation)
    }

    /// Changes the type of the annotation of this cluster, returning the old annotation.
    pub fn change_annotation<B>(self, annotation: B) -> (Cluster<T, B>, A) {
        let old_annotation = self.annotation;
        let cluster = Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children: self.children,
            annotation,
            parent_center_index: self.parent_center_index,
        };
        (cluster, old_annotation)
    }

    /// Changes the type of the annotation of this cluster using the provided function, returning the old annotation.
    ///
    /// The provided function is called before the annotation is updated, so it can use the current annotation if needed.
    pub fn change_annotation_with<F: Fn(&Self) -> B, B>(self, f: &F) -> (Cluster<T, B>, A) {
        let new_annotation = f(&self);
        self.change_annotation(new_annotation)
    }

    /// Compounds the annotation of this cluster with the additional annotation.
    ///
    /// The new annotation will be a tuple of the original annotation and the additional annotation. This can be useful for temporarily adding additional
    /// information to the cluster for use in specific algorithms without losing the original annotation. The original cluster type can be recovered using the
    /// [`decompound_annotation`](Cluster::decompound_annotation) method.
    pub fn compound_annotation<B>(self, additional_annotation: B) -> Cluster<T, (A, B)> {
        Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children: self.children,
            annotation: (self.annotation, additional_annotation),
            parent_center_index: self.parent_center_index,
        }
    }

    /// Compounds the annotation of this cluster using the provided function.
    pub fn compound_annotation_with<F: Fn(&Self) -> B, B>(self, f: &F) -> Cluster<T, (A, B)> {
        let additional_annotation = f(&self);
        self.compound_annotation(additional_annotation)
    }
}

impl<T, A, B> Cluster<T, (A, B)> {
    /// Removes the additional annotation from this cluster, returning a cluster with the original annotation and the additional annotation as a separate value.
    ///
    /// This is the inverse of the [`compound_annotation`](Cluster::compound_annotation) method.
    pub fn decompound_annotation(self) -> (Cluster<T, A>, B) {
        let (original_annotation, additional_annotation) = self.annotation;
        let cluster = Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children: self.children,
            annotation: original_annotation,
            parent_center_index: self.parent_center_index,
        };
        (cluster, additional_annotation)
    }
}

/// The number of features to include in the CSV export.
#[cfg(feature = "serde")]
const NUM_CLUSTER_FEATURES: usize = 8;

/// These methods, gated behind the `serde` feature, allow exporting the `Cluster` and its subtree to a CSV file.
#[cfg(feature = "serde")]
impl<T, A> Cluster<T, A>
where
    T: DistanceValue,
{
    /// Returns a CSV header string for the cluster information.
    #[must_use]
    pub const fn csv_header() -> [&'static str; NUM_CLUSTER_FEATURES] {
        [
            "depth",
            "center_index",
            "cardinality",
            "radius",
            "lfd",
            "num_children",
            "span",
            "parent_center_index",
        ]
    }

    /// Returns a row of CSV data representing the cluster's information.
    pub fn csv_row(&self) -> [String; NUM_CLUSTER_FEATURES] {
        [
            self.depth.to_string(),
            self.center_index.to_string(),
            self.cardinality().to_string(),
            self.radius().to_string(),
            self.lfd().to_string(),
            self.child_center_indices().map_or(0, <[_]>::len).to_string(),
            self.span().map_or_else(|| T::zero().to_string(), |s| s.to_string()),
            self.parent_center_index().unwrap_or(0).to_string(),
        ]
    }
}
