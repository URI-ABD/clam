//! Traits for adapting the `Ball` type to other types of `Cluster`s.

use distances::Number;

use crate::{dataset::ParDataset, metric::ParMetric, Ball, Dataset, Metric};

use super::{Adapted, ParParams, Params};

/// A trait for adapting a `Ball` into another `Cluster`.
///
/// # Type Parameters:
///
/// - `I`: The items in the `Dataset`.
/// - `T`: The type of the distance values.
/// - `Din`: The `Dataset` that the `Ball` was built on.
/// - `Dout`: The `Dataset` that the adapted `Cluster` will be built on.
/// - `M`: The `Metric` that the `Ball` was built with.
/// - `P`: The parameters used for adapting the `Ball`.
pub trait BallAdapter<I, T: Number, Din: Dataset<I>, Dout: Dataset<I>, M: Metric<I, T>, P: Params<I, T, Din, Ball<T>, M>>:
    Adapted<T, Ball<T>> + Sized
{
    /// Adapts this `Cluster` from a `Ball` tree.
    fn from_ball_tree(ball: Ball<T>, data: Din, metric: &M) -> (Self, Dout);
}

/// Parallel version of the [`BallAdapter`](BallAdapter) trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParBallAdapter<
    I: Send + Sync,
    T: Number,
    Din: ParDataset<I>,
    Dout: ParDataset<I>,
    M: ParMetric<I, T>,
    P: ParParams<I, T, Din, Ball<T>, M>,
>: Adapted<T, Ball<T>> + BallAdapter<I, T, Din, Dout, M, P> + Send + Sync
{
    /// Parallel version of [`BallAdapter::from_ball_tree`](BallAdapter::from_ball_tree).
    fn par_from_ball_tree(ball: Ball<T>, data: Din, metric: &M) -> (Self, Dout);
}
