//! The most basic representation of a `Cluster` is a metric-`Ball`.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use distances::Number;
use rayon::prelude::*;

use crate::{
    core::{dataset::ParDataset, metric::ParMetric, Dataset, Metric},
    utils,
};

use super::{partition::ParPartition, Cluster, ParCluster, Partition, LFD};

/// A metric-`Ball` is a collection of items that are within a certain distance
/// of a center.
///
/// # Example
///
/// ```rust
/// use abd_clam::{
///     cluster::{Partition, ParPartition},
///     metric::AbsoluteDifference,
///     Ball, Cluster, Dataset, FlatVec
/// };
///
/// let items = (0..=100).collect::<Vec<_>>();
/// let data = FlatVec::new(items).unwrap();
/// let metric = AbsoluteDifference;
///
/// // We will create a `Ball` with all the items in the `data`.
/// let indices = data.indices().collect::<Vec<_>>();
/// let ball = Ball::new(&data, &metric, &indices, 0, None).unwrap();
///
/// assert_eq!(ball.depth(), 0);
/// assert_eq!(ball.cardinality(), 101);
/// assert_eq!(ball.indices(), indices);
/// assert_eq!(ball.arg_center(), 50);
/// assert_eq!(ball.radius(), 50);
/// assert!([0, 100].contains(&ball.arg_radial()));
/// assert!((ball.lfd() - 1.0).abs() < 0.1);
///
/// assert!(ball.is_leaf());
///
/// // We will now create a tree of `Ball`s with leaves being singletons.
/// let partition_criteria = |ball: &Ball<_>| ball.cardinality() > 1;
/// let root = Ball::new_tree(&data, &metric, &partition_criteria, None);
/// assert!(!root.is_leaf());
///
/// // We can also use the equivalent parallelized methods to create the tree.
/// let root = Ball::par_new_tree(&data, &metric, &partition_criteria, None);
/// assert!(!root.is_leaf());
/// ```
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)
)]
#[cfg_attr(feature = "disk-io", bitcode(recursive))]
pub struct Ball<T: Number> {
    /// Parameters used for creating the `Ball`.
    depth: usize,
    /// The number of items in the `Ball`.
    cardinality: usize,
    /// The radius of the `Ball`.
    radius: T,
    /// The local fractal dimension of the `Ball`.
    lfd: f32,
    /// The index of the center item.
    arg_center: usize,
    /// The index of the item that is the furthest from the center.
    arg_radial: usize,
    /// The indices of the items in the `Ball`.
    indices: Vec<usize>,
    /// The extents of the `Ball`.
    extents: Vec<(usize, T)>,
    /// The children of the `Ball`.
    children: Vec<Box<Self>>,
}

impl<T: Number> core::fmt::Debug for Ball<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ball")
            .field("depth", &self.depth)
            .field("cardinality", &self.cardinality)
            .field("radius", &self.radius)
            .field("lfd", &self.lfd)
            .field("arg_center", &self.arg_center)
            .field("arg_radial", &self.arg_radial)
            .field("indices", &self.indices)
            .field("extents", &self.extents)
            .field("children", &!self.children.is_empty())
            .finish()
    }
}

impl<T: Number> PartialEq for Ball<T> {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth && self.cardinality == other.cardinality && self.indices == other.indices
    }
}

impl<T: Number> Eq for Ball<T> {}

impl<T: Number> PartialOrd for Ball<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Number> Ord for Ball<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.depth
            .cmp(&other.depth)
            .then_with(|| self.cardinality.cmp(&other.cardinality))
            .then_with(|| self.indices.cmp(&other.indices))
    }
}

impl<T: Number> Hash for Ball<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // We hash the `indices` field
        self.indices.hash(state);
    }
}

impl<T: Number> Cluster<T> for Ball<T> {
    fn depth(&self) -> usize {
        self.depth
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn arg_center(&self) -> usize {
        self.arg_center
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.arg_center = arg_center;
    }

    fn radius(&self) -> T {
        self.radius
    }

    fn arg_radial(&self) -> usize {
        self.arg_radial
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.arg_radial = arg_radial;
    }

    fn lfd(&self) -> f32 {
        self.lfd
    }

    fn contains(&self, index: usize) -> bool {
        self.indices.contains(&index)
    }

    fn indices(&self) -> Vec<usize> {
        self.indices.clone()
    }

    fn set_indices(&mut self, indices: &[usize]) {
        self.indices = indices.to_vec();
    }

    fn extents(&self) -> &[(usize, T)] {
        &self.extents
    }

    fn extents_mut(&mut self) -> &mut [(usize, T)] {
        &mut self.extents
    }

    fn add_extent(&mut self, index: usize, extent: T) {
        self.extents.push((index, extent));
    }

    fn take_extents(&mut self) -> Vec<(usize, T)> {
        core::mem::take(&mut self.extents)
    }

    fn children(&self) -> Vec<&Self> {
        self.children.iter().map(AsRef::as_ref).collect()
    }

    fn children_mut(&mut self) -> Vec<&mut Self> {
        self.children.iter_mut().map(AsMut::as_mut).collect()
    }

    fn set_children(&mut self, children: Vec<Box<Self>>) {
        self.children = children;
    }

    fn take_children(&mut self) -> Vec<Box<Self>> {
        core::mem::take(&mut self.children)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.indices.iter().all(|i| other.indices.contains(i))
    }
}

impl<T: Number> ParCluster<T> for Ball<T> {
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        self.indices.par_iter().copied()
    }
}

impl<T: Number> Partition<T> for Ball<T> {
    fn new<I, D: Dataset<I>, M: Metric<I, T>>(
        data: &D,
        metric: &M,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if indices.is_empty() {
            return Err("Cannot create a Ball with no items".to_string());
        }

        let cardinality = indices.len();
        let samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            let num_samples = utils::num_samples(cardinality, 100, 10_000);
            data.choose_unique(indices, num_samples, seed, metric)
        };

        let arg_center = data.median(&samples, metric);

        let distances = data.one_to_many(arg_center, indices, metric).collect::<Vec<_>>();
        let &(arg_radial, radius) = distances
            .iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        let distances = distances.into_iter().map(|(_, d)| d).collect::<Vec<_>>();
        let lfd = LFD::from_radial_distances(&distances, radius.half());

        Ok(Self {
            depth,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            indices: indices.to_vec(),
            extents: vec![(arg_center, radius)],
            children: Vec::new(),
        })
    }

    fn find_extrema<I, D: Dataset<I>, M: Metric<I, T>>(&mut self, data: &D, metric: &M) -> Vec<usize> {
        let (arg_l, d) = data
            .one_to_many(self.arg_radial, &self.indices, metric)
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        self.add_extent(self.arg_radial, d);

        vec![arg_l, self.arg_radial]
    }
}

impl<T: Number> ParPartition<T> for Ball<T> {
    fn par_new<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(
        data: &D,
        metric: &M,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if indices.is_empty() {
            return Err("Cannot create a Ball with no items".to_string());
        }

        let cardinality = indices.len();
        let samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            let num_samples = utils::num_samples(cardinality, 100, 10_000);
            data.choose_unique(indices, num_samples, seed, metric)
        };

        let arg_center = data.par_median(&samples, metric);

        let distances = data.par_one_to_many(arg_center, indices, metric).collect::<Vec<_>>();
        let &(arg_radial, radius) = distances
            .iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        let distances = distances.into_iter().map(|(_, d)| d).collect::<Vec<_>>();
        let lfd = LFD::from_radial_distances(&distances, radius.half());

        Ok(Self {
            depth,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            indices: indices.to_vec(),
            extents: vec![(arg_center, radius)],
            children: Vec::new(),
        })
    }

    fn par_find_extrema<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(
        &mut self,
        data: &D,
        metric: &M,
    ) -> Vec<usize> {
        let (arg_l, d) = data
            .par_one_to_many(self.arg_radial, &self.indices, metric)
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        self.add_extent(self.arg_radial, d);

        vec![arg_l, self.arg_radial]
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number> super::Csv<T> for Ball<T> {
    fn header(&self) -> Vec<String> {
        vec![
            "depth".to_string(),
            "cardinality".to_string(),
            "radius".to_string(),
            "lfd".to_string(),
            "arg_center".to_string(),
            "arg_radial".to_string(),
            "is_leaf".to_string(),
        ]
    }

    fn row(&self) -> Vec<String> {
        vec![
            self.depth.to_string(),
            self.cardinality.to_string(),
            self.radius.to_string(),
            format!("{:.8}", self.lfd),
            self.arg_center.to_string(),
            self.arg_radial.to_string(),
            self.children.is_empty().to_string(),
        ]
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number> super::ParCsv<T> for Ball<T> {}

#[cfg(feature = "disk-io")]
impl<T: Number> super::ClusterIO<T> for Ball<T> {}

#[cfg(feature = "disk-io")]
impl<T: Number> super::ParClusterIO<T> for Ball<T> {}
