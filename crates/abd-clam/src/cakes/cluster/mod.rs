//! An adaptation of `Ball` that stores indices after reordering the dataset.

mod offset_ball;
mod searchable;
mod squishy_ball;

use distances::Number;
pub use offset_ball::OffBall;
pub use searchable::{ParSearchable, Searchable};
pub use squishy_ball::SquishyBall;

use crate::{cluster::ParCluster, dataset::ParDataset, Ball, Cluster, Dataset};

impl<I, U: Number, D: Dataset<I, U>> Searchable<I, U, D> for Ball<U> {}
impl<I, U: Number, D: Dataset<I, U>, S: Cluster<U>> Searchable<I, U, D> for OffBall<U, S> {}
impl<I, U: Number, D: Dataset<I, U>, S: Cluster<U>> Searchable<I, U, D> for SquishyBall<U, S> {}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParSearchable<I, U, D> for Ball<U> {}
impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<U>> ParSearchable<I, U, D> for OffBall<U, S> {}
impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<U>> ParSearchable<I, U, D> for SquishyBall<U, S> {}
