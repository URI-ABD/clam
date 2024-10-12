//! The `Vertex` is a `Cluster` adapter used to represent `Cluster`s in the
//! `Graph`.

use core::marker::PhantomData;

use distances::Number;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    adapter::{Adapter, BallAdapter, ParAdapter, ParBallAdapter, ParParams, Params},
    chaoda::NUM_RATIOS,
    cluster::ParCluster,
    dataset::ParDataset,
    Ball, Cluster, Dataset,
};

/// The `Vertex` is a `Cluster` adapter used to represent `Cluster`s in the
/// `Graph`.
///
/// # Type Parameters
///
/// - `I`: The type on instances in the dataset.
/// - `U`: The type of the distance values.
/// - `D`: The type of the dataset.
/// - `S`: The type of the `Cluster` that was adapted into the `Vertex`.
#[derive(Serialize, Deserialize)]
pub struct Vertex<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The `Cluster` that was adapted into the `Vertex`.
    source: S,
    /// The children of the `Vertex`.
    children: Vec<(usize, U, Box<Self>)>,
    /// The anomaly detection properties of the `Vertex`.
    params: Ratios,
    /// Phantom data to satisfy the compiler.
    _id: PhantomData<(I, D)>,
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D> + std::fmt::Debug> std::fmt::Debug for Vertex<I, U, D, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Vertex")
            .field("source", &self.source)
            .field("children", &self.children)
            .field("ratios", &self.params.ratios)
            .field("ema_ratios", &self.params.ema_ratios)
            .field("accumulated_cp_car_ratio", &self.params.accumulated_cp_car_ratio)
            .finish()
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Vertex<I, U, D, S> {
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

impl<I, U: Number, D: Dataset<I, U>> BallAdapter<I, U, D, D, Ratios> for Vertex<I, U, D, Ball<I, U, D>> {
    /// Creates a new `OffsetBall` tree from a `Ball` tree.
    fn from_ball_tree(ball: Ball<I, U, D>, data: D) -> (Self, D) {
        let root = Self::adapt_tree(ball, None);
        (root, data)
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParBallAdapter<I, U, D, D, Ratios>
    for Vertex<I, U, D, Ball<I, U, D>>
{
    /// Creates a new `OffsetBall` tree from a `Ball` tree.
    fn par_from_ball_tree(ball: Ball<I, U, D>, data: D) -> (Self, D) {
        let root = Self::par_adapt_tree(ball, None);
        (root, data)
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Adapter<I, U, D, D, S, Ratios> for Vertex<I, U, D, S> {
    fn new_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, params: Ratios) -> Self {
        Self {
            source,
            children,
            params,
            _id: PhantomData,
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

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParAdapter<I, U, D, D, S, Ratios>
    for Vertex<I, U, D, S>
{
    fn par_post_traversal(&mut self) {}
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Cluster<I, U, D> for Vertex<I, U, D, S> {
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

    fn radius(&self) -> U {
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

    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.source.indices()
    }

    fn set_indices(&mut self, indices: Vec<usize>) {
        self.source.set_indices(indices);
    }

    fn children(&self) -> &[(usize, U, Box<Self>)] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [(usize, U, Box<Self>)] {
        &mut self.children
    }

    fn set_children(&mut self, children: Vec<(usize, U, Box<Self>)>) {
        self.children = children;
    }

    fn take_children(&mut self) -> Vec<(usize, U, Box<Self>)> {
        core::mem::take(&mut self.children)
    }

    fn distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        self.source.distances_to_query(data, query)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.source.is_descendant_of(&other.source)
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParCluster<I, U, D> for Vertex<I, U, D, S> {
    fn par_distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        self.source.par_distances_to_query(data, query)
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialEq for Vertex<I, U, D, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Eq for Vertex<I, U, D, S> {}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialOrd for Vertex<I, U, D, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Ord for Vertex<I, U, D, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> std::hash::Hash for Vertex<I, U, D, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

/// The anomaly detection properties of the `Vertex`, their exponential moving
/// averages, and the accumulated child-parent cardinality ratio.
#[allow(clippy::struct_field_names)]
#[derive(Serialize, Deserialize)]
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

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Params<I, U, D, D, S> for Ratios {
    fn child_params(&self, children: &[S]) -> Vec<Self> {
        children.iter().map(|child| child_params(self, child)).collect()
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParParams<I, U, D, D, S> for Ratios {
    fn par_child_params(&self, children: &[S]) -> Vec<Self> {
        children.par_iter().map(|child| child_params(self, child)).collect()
    }
}

/// Computes the anomaly detection properties of a child `Cluster` given the
/// anomaly detection properties of the parent `Cluster`.
#[allow(clippy::similar_names)]
fn child_params<I, U: Number, D: Dataset<I, U>, C: Cluster<I, U, D>>(parent: &Ratios, child: &C) -> Ratios {
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
