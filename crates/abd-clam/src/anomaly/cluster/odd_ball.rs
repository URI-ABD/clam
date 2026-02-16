//! The `OddBall` is a `Cluster` adapter used to represent `Cluster`s in the
//! `Graph`.

use rayon::prelude::*;

use crate::{
    chaoda::{ParVertex, Vertex},
    Cluster, DistanceValue, ParCluster,
};

/// The `OddBall` is a `Cluster` adapter used to represent `Cluster`s in the
/// `Graph`.
///
/// # Type Parameters
///
/// - `T`: The type of the distance values.
/// - `S`: The type of the `Cluster` that was adapted into the `OddBall`.
#[derive(Clone, bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)]
#[bitcode(recursive)]
pub struct OddBall<T: DistanceValue, S: Cluster<T>> {
    /// The `Cluster` that was adapted into the `OddBall`.
    pub(crate) source: S,
    /// The children of the `OddBall`.
    children: Vec<Box<Self>>,
    /// The anomaly detection properties of the `OddBall`.
    ratios: [f32; 6],
    /// The accumulated parent-child cardinality ratio.
    accumulated_pc_cardinality_ratio: f32,
    /// Ghost in the machine
    marker: std::marker::PhantomData<T>,
}

impl<T: DistanceValue, S: Cluster<T>> OddBall<T, S> {
    /// Creates a new `OddBall` tree from a source `Cluster` tree and reorders
    /// the dataset in place.
    pub fn from_cluster_tree(root: S) -> Self {
        Self::adapt_tree_recursive(root, [1.0; 6], 1.0).0
    }

    /// Recursive helper for [`from_cluster_tree`](Self::from_cluster_tree).
    fn adapt_tree_recursive(
        mut source: S,
        ratios: [f32; 6],
        accumulated_pc_cardinality_ratio: f32,
    ) -> (Self, Vec<usize>) {
        let (children, indices) = if source.is_leaf() {
            (vec![], source.take_indices())
        } else {
            let children = source.take_children();

            let (children, child_indices): (Vec<_>, Vec<_>) = children
                .into_iter()
                .map(|child| {
                    let (c_ratios, c_acc) = child_params(ratios, accumulated_pc_cardinality_ratio, child.as_ref());
                    Self::adapt_tree_recursive(*child, c_ratios, c_acc)
                })
                .map(|(c, indices)| (Box::new(c), indices))
                .unzip();

            let indices = child_indices.into_iter().flatten().collect();

            (children, indices)
        };

        let c = Self {
            source,
            children,
            ratios,
            accumulated_pc_cardinality_ratio,
            marker: core::marker::PhantomData,
        };
        (c, indices)
    }
}

/// Computes the anomaly detection properties of a child `Cluster` given the
/// anomaly detection properties of the parent `Cluster`.
#[allow(clippy::similar_names)]
fn child_params<T: DistanceValue, C: Cluster<T>>(parent_ratios: [f32; 6], p_acc: f32, child: &C) -> ([f32; 6], f32) {
    let [pc, pr, pl, pc_, pr_, pl_] = parent_ratios;

    let c = child.cardinality() as f32 / pc;
    let r = child
        .radius()
        .to_f32()
        .unwrap_or_else(|| unreachable!("Could not convert radius to f32"))
        / pr;
    let l = child.lfd() / pl;

    let c_ = crate::utils::next_ema(c, pc_);
    let r_ = crate::utils::next_ema(r, pr_);
    let l_ = crate::utils::next_ema(l, pl_);

    ([c, r, l, c_, r_, l_], p_acc + c)
}

impl<T: DistanceValue + core::fmt::Debug, S: Cluster<T> + core::fmt::Debug> core::fmt::Debug for OddBall<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("OddBall")
            .field("source", &self.source)
            .field("ratios", &self.ratios)
            .field(
                "accumulated_pc_cardinality_ratio",
                &self.accumulated_pc_cardinality_ratio,
            )
            .field("children", &!self.children.is_empty())
            .finish()
    }
}

impl<T: DistanceValue, S: Cluster<T> + PartialEq> PartialEq for OddBall<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<T: DistanceValue, S: Cluster<T> + Eq> Eq for OddBall<T, S> {}

impl<T: DistanceValue, S: Cluster<T> + PartialOrd> PartialOrd for OddBall<T, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: DistanceValue, S: Cluster<T> + Ord> Ord for OddBall<T, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<T: DistanceValue, S: Cluster<T>> std::hash::Hash for OddBall<T, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let id = (self.source.arg_center(), self.source.cardinality());
        id.hash(state);
    }
}

impl<T: DistanceValue, S: Cluster<T>> Cluster<T> for OddBall<T, S> {
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

    fn contains(&self, idx: usize) -> bool {
        self.source.contains(idx)
    }

    fn indices(&self) -> Vec<usize> {
        self.source.indices()
    }

    fn set_indices(&mut self, indices: &[usize]) {
        self.source.set_indices(indices);
    }

    fn take_indices(&mut self) -> Vec<usize> {
        self.source.take_indices()
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
        std::mem::take(&mut self.children)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.source.is_descendant_of(&other.source)
    }
}

impl<T: DistanceValue + Send + Sync, S: ParCluster<T>> ParCluster<T> for OddBall<T, S> {
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        self.source.par_indices()
    }
}

impl<T: DistanceValue, S: Cluster<T>> Vertex<T> for OddBall<T, S> {
    const NUM_FEATURES: usize = 6;

    type FeatureVector = [f32; 6];

    fn feature_vector(&self) -> Self::FeatureVector {
        self.ratios
    }

    fn accumulated_cp_cardinality_ratio(&self) -> f32 {
        self.accumulated_pc_cardinality_ratio
    }
}

impl<T: DistanceValue + Send + Sync, S: ParCluster<T>> ParVertex<T> for OddBall<T, S> {}
