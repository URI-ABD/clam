//! A `Vertex` to use as a node in a `Graph`.

use distances::Number;

use crate::{Adapter, Ball, Children, Cluster, Params};

use super::OddBall;

/// The ratios of the `Vertex` and their exponential moving averages.
pub type Ratios = [f32; 6];

/// A `Vertex` to use as a node in a `Graph`.
#[derive(Debug)]
pub struct Vertex<U: Number> {
    /// The `Ball` that was adapted into this `Vertex`.
    ball: Ball<U>,
    /// The children of the `Vertex`.
    children: Option<Children<U, Self>>,
    /// The parameters to use for the `Vertex`.
    params: VertexParams,
}

impl<U: Number> Vertex<U> {
    /// Normalizes the ratios of the tree of `Vertex`s.
    #[must_use]
    pub fn with_normalized_ratios(mut self) -> Self {
        let all_ratios = self.subtree().into_iter().map(Self::ratios_array).collect::<Vec<_>>();
        let all_ratios = crate::utils::rows_to_cols(&all_ratios);

        let means = crate::utils::calc_row_means(&all_ratios);
        let sds = crate::utils::calc_row_sds(&all_ratios);

        self.set_normalized_ratios(means, sds);

        self
    }

    /// Returns the ratios of the `Vertex` along with the exponential moving averages.
    const fn ratios_array(&self) -> Ratios {
        let [c, r, l] = self.params.ratios;
        let [c_, r_, l_] = self.params.ema_ratios;
        [c, r, l, c_, r_, l_]
    }

    /// Recursively applies Gaussian error normalization to the ratios in the subtree.
    fn set_normalized_ratios(&mut self, means: Ratios, sds: Ratios) {
        let normalized_ratios = self
            .ratios_array()
            .into_iter()
            .zip(means)
            .zip(sds)
            .map(|((value, mean), std)| (value - mean) / std.mul_add(core::f32::consts::SQRT_2, f32::EPSILON))
            .map(libm::erff)
            .map(|v| (1. + v) / 2.)
            .collect::<Vec<_>>();

        let [c, r, l, c_, r_, l_] = normalized_ratios
            .try_into()
            .unwrap_or_else(|e| unreachable!("Ratios array has invalid length. {e:?}"));
        self.params.ratios = [c, r, l];
        self.params.ema_ratios = [c_, r_, l_];

        if let Some(children) = self.children.as_mut() {
            children
                .clusters_mut()
                .iter_mut()
                .for_each(|c| c.set_normalized_ratios(means, sds));
        };
    }
}

impl<U: Number> OddBall<U> for Vertex<U> {
    fn ratios(&self) -> Vec<f32> {
        self.ratios_array().to_vec()
    }

    fn accumulated_cp_car_ratio(&self) -> f32 {
        self.params.accumulated_cp_car_ratio
    }
}

/// The parameters to use for the `Vertex`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::module_name_repetitions)]
pub struct VertexParams {
    /// The anomaly detection properties of the `Vertex`.
    ratios: [f32; 3],
    /// The exponential moving averages of `ratios`.
    ema_ratios: [f32; 3],
    /// The accumulated child-parent cardinality ratio.
    accumulated_cp_car_ratio: f32,
}

impl Default for VertexParams {
    fn default() -> Self {
        Self {
            ratios: [1.0; 3],
            ema_ratios: [1.0; 3],
            accumulated_cp_car_ratio: 1.0,
        }
    }
}

impl<U: Number> Params<U> for VertexParams {
    #[allow(clippy::similar_names)]
    fn child_params(&self, child: &Ball<U>) -> Self {
        let [pc, pr, pl] = self.ratios;
        let c = child.cardinality().as_f32() / pc;
        let r = child.radius().as_f32() / pr;
        let l = child.lfd().as_f32() / pl;
        let ratios = [c, r, l];

        let [pc_, pr_, pl_] = self.ema_ratios;
        let c_ = crate::utils::next_ema(c, pc_);
        let r_ = crate::utils::next_ema(r, pr_);
        let l_ = crate::utils::next_ema(l, pl_);
        let ema_ratios = [c_, r_, l_];

        let accumulated_cp_car_ratio = self.accumulated_cp_car_ratio + c;

        Self {
            ratios,
            ema_ratios,
            accumulated_cp_car_ratio,
        }
    }
}

impl<U: Number> Adapter<U, VertexParams> for Vertex<U> {
    fn adapt_one(ball: Ball<U>, params: VertexParams) -> Self
    where
        Self: Sized,
    {
        Self {
            ball,
            children: None,
            params,
        }
    }

    fn ball(&self) -> &Ball<U> {
        &self.ball
    }

    fn ball_mut(&mut self) -> &mut Ball<U> {
        &mut self.ball
    }
}

impl<U: Number> Cluster<U> for Vertex<U> {
    fn new<I, D: crate::Dataset<I, U>>(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize)
    where
        Self: Sized,
    {
        let (ball, arg_radial) = Ball::new(data, indices, depth, seed);
        let vertex = Self {
            ball,
            children: None,
            params: VertexParams::default(),
        };
        (vertex, arg_radial)
    }

    fn depth(&self) -> usize {
        self.ball.depth()
    }

    fn cardinality(&self) -> usize {
        self.ball.cardinality()
    }

    fn arg_center(&self) -> usize {
        self.ball.arg_center()
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.ball.set_arg_center(arg_center);
    }

    fn radius(&self) -> U {
        self.ball.radius()
    }

    fn arg_radial(&self) -> usize {
        self.ball.arg_radial()
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.ball.set_arg_radial(arg_radial);
    }

    fn lfd(&self) -> f32 {
        self.ball.lfd()
    }

    fn index_store(&self) -> &crate::core::cluster::IndexStore {
        self.ball.index_store()
    }

    fn set_index_store(&mut self, indices: crate::core::cluster::IndexStore) {
        self.ball.set_index_store(indices);
    }

    fn children(&self) -> Option<&Children<U, Self>>
    where
        Self: Sized,
    {
        self.children.as_ref()
    }

    fn children_mut(&mut self) -> Option<&mut Children<U, Self>>
    where
        Self: Sized,
    {
        self.children.as_mut()
    }

    fn set_children(self, children: Children<U, Self>) -> Self
    where
        Self: Sized,
    {
        Self {
            ball: self.ball,
            children: Some(children),
            params: self.params,
        }
    }

    fn find_extrema<I, D: crate::Dataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>) {
        self.ball.find_extrema(data)
    }
}

impl<U: Number> PartialEq for Vertex<U> {
    fn eq(&self, other: &Self) -> bool {
        self.ball == other.ball
    }
}

impl<U: Number> PartialOrd for Vertex<U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.ball.partial_cmp(&other.ball)
    }
}
