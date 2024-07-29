//! An adaptation of `Ball` that stores indices after reordering the dataset.

mod offset_ball;
mod searchable;

use distances::Number;
pub use offset_ball::OffsetBall;
pub use searchable::{ParSearchable, Searchable};

use crate::{dataset::ParDataset, Ball, Dataset};

impl<I, U: Number, D: Dataset<I, U>> Searchable<I, U, D> for Ball<U> {}
impl<I, U: Number, D: Dataset<I, U>> Searchable<I, U, D> for OffsetBall<U> {}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParSearchable<I, U, D> for Ball<U> {}
impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParSearchable<I, U, D> for OffsetBall<U> {}
