//! The `Vertex` is a `Cluster` adapter used to represent `Cluster`s in the
//! `Graph`.

use distances::Number;
use rayon::prelude::*;

use crate::{
    chaoda::NUM_RATIOS,
    cluster::{
        adapter::{Adapter, BallAdapter, ParAdapter, ParBallAdapter, ParParams, Params},
        ParCluster,
    },
    dataset::ParDataset,
    metric::ParMetric,
    Ball, Cluster, Dataset, Metric,
};

/// The `Vertex` is a `Cluster` adapter used to represent `Cluster`s in the
/// `Graph`.
///
/// # Type Parameters
///
/// - `T`: The type of the distance values.
/// - `S`: The type of the `Cluster` that was adapted into the `Vertex`.
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "disk-io", bitcode(recursive))]
pub struct Vertex<T: Number, S: Cluster<T>> {
    /// The `Cluster` that was adapted into the `Vertex`.
    pub(crate) source: S,
    /// The children of the `Vertex`.
    children: Vec<Box<Self>>,
    /// The anomaly detection properties of the `Vertex`.
    params: Ratios,
    /// Ghosts in the machine.
    phantom: core::marker::PhantomData<T>,
}

impl<T: Number, S: Cluster<T> + core::fmt::Debug> core::fmt::Debug for Vertex<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Vertex")
            .field("source", &self.source)
            .field("ratios", &self.params.ratios)
            .field("ema_ratios", &self.params.ema_ratios)
            .field("accumulated_cp_car_ratio", &self.params.accumulated_cp_car_ratio)
            .field("children", &!self.children.is_empty())
            .finish()
    }
}

impl<T: Number, S: Cluster<T>> PartialEq for Vertex<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<T: Number, S: Cluster<T>> Eq for Vertex<T, S> {}

impl<T: Number, S: Cluster<T>> PartialOrd for Vertex<T, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Number, S: Cluster<T>> Ord for Vertex<T, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<T: Number, S: Cluster<T>> core::hash::Hash for Vertex<T, S> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

impl<T: Number, S: Cluster<T>> Vertex<T, S> {
    /// Returns the anomaly detection properties of the `Vertex` and their
    /// exponential moving averages.
    #[must_use]
    pub const fn ratios(&self) -> [f32; NUM_RATIOS] {
        let [c, r, l] = self.params.ratios;
        let [c_, r_, l_] = self.params.ema_ratios;
        [c, r, l, c_, r_, l_]
    }

    /// Returns the accumulated child-parent cardinality ratio.
    #[must_use]
    pub const fn accumulated_cp_car_ratio(&self) -> f32 {
        self.params.accumulated_cp_car_ratio
    }
}

impl<T: Number, S: Cluster<T>> Cluster<T> for Vertex<T, S> {
    fn depth(&self) -> usize {
        self.source.depth()
    }

    fn cardinality(&self) -> usize {
        self.source.cardinality()
    }

    fn arg_center(&self) -> usize {
        self.source.arg_center()
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.source.set_arg_center(arg_center);
    }

    fn radius(&self) -> T {
        self.source.radius()
    }

    fn arg_radial(&self) -> usize {
        self.source.arg_radial()
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.source.set_arg_radial(arg_radial);
    }

    fn lfd(&self) -> f32 {
        self.source.lfd()
    }

    fn contains(&self, index: usize) -> bool {
        self.source.contains(index)
    }

    fn indices(&self) -> Vec<usize> {
        self.source.indices()
    }

    fn set_indices(&mut self, indices: &[usize]) {
        self.source.set_indices(indices);
    }

    fn extents(&self) -> &[(usize, T)] {
        self.source.extents()
    }

    fn extents_mut(&mut self) -> &mut [(usize, T)] {
        self.source.extents_mut()
    }

    fn add_extent(&mut self, idx: usize, extent: T) {
        self.source.add_extent(idx, extent);
    }

    fn take_extents(&mut self) -> Vec<(usize, T)> {
        self.source.take_extents()
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
        self.source.is_descendant_of(&other.source)
    }
}

impl<T: Number, S: ParCluster<T>> ParCluster<T> for Vertex<T, S> {
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        self.source.par_indices()
    }
}

/// The anomaly detection properties of the `Vertex`, their exponential moving
/// averages, and the accumulated child-parent cardinality ratio.
#[allow(clippy::struct_field_names)]
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
pub struct Ratios {
    /// The anomaly detection properties of the `Vertex`.
    ratios: [f32; 3],
    /// The exponential moving average of the `ratios`.
    ema_ratios: [f32; 3],
    /// The accumulated child-parent cardinality ratio.
    accumulated_cp_car_ratio: f32,
}

impl Default for Ratios {
    fn default() -> Self {
        Self {
            ratios: [1.0; 3],
            ema_ratios: [1.0; 3],
            accumulated_cp_car_ratio: 1.0,
        }
    }
}

impl<I, T: Number, D: Dataset<I>, S: Cluster<T>> Params<I, T, D, S> for Ratios {
    fn child_params<M: Metric<I, T>>(&self, children: &[S], _: &D, _: &M) -> Vec<Self> {
        children.iter().map(|child| child_params(self, child)).collect()
    }
}

impl<I: Send + Sync, T: Number, D: ParDataset<I>, S: ParCluster<T>> ParParams<I, T, D, S> for Ratios {
    fn par_child_params<M: ParMetric<I, T>>(&self, children: &[S], _: &D, _: &M) -> Vec<Self> {
        children.par_iter().map(|child| child_params(self, child)).collect()
    }
}

/// Computes the anomaly detection properties of a child `Cluster` given the
/// anomaly detection properties of the parent `Cluster`.
#[allow(clippy::similar_names)]
fn child_params<T: Number, C: Cluster<T>>(parent: &Ratios, child: &C) -> Ratios {
    let [pc, pr, pl] = parent.ratios;
    let c = child.cardinality().as_f32() / pc;
    let r = child.radius().as_f32() / pr;
    let l = child.lfd().as_f32() / pl;
    let ratios = [c, r, l];

    let [pc_, pr_, pl_] = parent.ema_ratios;
    let c_ = crate::utils::next_ema(c, pc_);
    let r_ = crate::utils::next_ema(r, pr_);
    let l_ = crate::utils::next_ema(l, pl_);

    let accumulated_cp_car_ratio = parent.accumulated_cp_car_ratio + c;

    Ratios {
        ratios,
        ema_ratios: [c_, r_, l_],
        accumulated_cp_car_ratio,
    }
}

impl<I, T: Number, D: Dataset<I>> BallAdapter<I, T, D, D, Ratios> for Vertex<T, Ball<T>> {
    /// Creates a new `OffsetBall` tree from a `Ball` tree.
    fn from_ball_tree<M: Metric<I, T>>(ball: Ball<T>, data: D, metric: &M) -> (Self, D) {
        let root = Self::adapt_tree(ball, None, &data, metric);
        (root, data)
    }
}

impl<I: Send + Sync, T: Number, D: ParDataset<I>> ParBallAdapter<I, T, D, D, Ratios> for Vertex<T, Ball<T>> {
    /// Creates a new `OffsetBall` tree from a `Ball` tree.
    fn par_from_ball_tree<M: ParMetric<I, T>>(ball: Ball<T>, data: D, metric: &M) -> (Self, D) {
        let root = Self::par_adapt_tree(ball, None, &data, metric);
        (root, data)
    }
}

impl<I, T: Number, D: Dataset<I>, S: Cluster<T>> Adapter<I, T, D, D, S, Ratios> for Vertex<T, S> {
    fn new_adapted<M: Metric<I, T>>(source: S, children: Vec<Box<Self>>, params: Ratios, _: &D, _: &M) -> Self {
        Self {
            source,
            params,
            children,
            phantom: core::marker::PhantomData,
        }
    }

    fn post_traversal(&mut self) {}

    fn source(&self) -> &S {
        &self.source
    }

    fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    fn take_source(self) -> S {
        self.source
    }

    fn params(&self) -> &Ratios {
        &self.params
    }
}

impl<I: Send + Sync, T: Number, D: ParDataset<I>, S: ParCluster<T>> ParAdapter<I, T, D, D, S, Ratios> for Vertex<T, S> {
    fn par_new_adapted<M: ParMetric<I, T>>(
        source: S,
        children: Vec<Box<Self>>,
        params: Ratios,
        data: &D,
        metric: &M,
    ) -> Self {
        Self::new_adapted(source, children, params, data, metric)
    }
}
