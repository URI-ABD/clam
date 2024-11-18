//! An adaptation of `Ball` that stores indices after reordering the dataset.

use distances::Number;
pub use offset_ball::OffBall;
pub use searchable::{ParSearchable, Searchable};

use crate::{cluster::ParCluster, dataset::ParDataset, Ball, Cluster, Dataset};

mod offset_ball;
mod searchable;

impl<I, U: Number, D: Dataset<I, U>> Searchable<I, U, D> for Ball<I, U, D> {}
impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Searchable<I, U, D> for OffBall<I, U, D, S> {}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParSearchable<I, U, D> for Ball<I, U, D> {}
impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParSearchable<I, U, D>
    for OffBall<I, U, D, S>
{
}
