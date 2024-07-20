//! A `Dataset` that can be sharded into multiple smaller datasets.

use distances::Number;

use crate::new_core::{Dataset, FlatVec, Metric};

/// A dataset that can be sharded into multiple smaller datasets.
pub trait Shardable<I, U: Number>: Dataset<I, U> {
    /// Sets the permutation of the dataset to the identity permutation.
    ///
    /// This should not change the order of the instances.
    #[must_use]
    fn reset_permutation(self) -> Self
    where
        Self: Sized;

    /// Split the dataset into two smaller datasets, at the given index.
    ///
    /// If the `Dataset` is `Permutable`, the permutation should be ignored.
    ///
    /// # Arguments
    ///
    /// * `at` - The index at which to split the dataset.
    ///
    /// # Returns
    ///
    /// - The dataset containing instances in the range `[0, at)`.
    /// - The dataset containing instances in the range `[at, cardinality)`.
    fn split_off(self, at: usize) -> [Self; 2]
    where
        Self: Sized;

    /// Shard the dataset into a number of smaller datasets.
    ///
    /// This will erase the permutation of the dataset.
    ///
    /// # Arguments
    ///
    /// * `at` - The indices at which to shard the dataset.
    ///
    /// # Returns
    ///
    /// A vector of datasets, each containing instances in the range `[at[i], at[i+1])`.
    fn shard(mut self, at: &[usize]) -> Vec<Self>
    where
        Self: Sized,
    {
        let mut shards = Vec::with_capacity(at.len() + 1);
        // Iterate over `at` in reverse order
        for &i in at.iter().rev() {
            let [left, right] = self.split_off(i);
            shards.push(right);
            self = left;
        }
        shards.push(self);
        shards.reverse();
        shards.into_iter().map(Self::reset_permutation).collect()
    }

    /// Shard the dataset into a number of smaller datasets, with each shard
    /// containing an equal number of instances, except possibly the last shard.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of instances in each shard.
    ///
    /// # Returns
    ///
    /// A vector of datasets, each containing `size` instances, except possibly
    /// the last shard.
    fn shard_evenly(self, size: usize) -> Vec<Self>
    where
        Self: Sized,
    {
        let at = (0..self.cardinality()).step_by(size).collect::<Vec<_>>();
        self.shard(&at)
    }
}

impl<I, U: Number, M> Shardable<I, U> for FlatVec<I, U, M> {
    #[must_use]
    fn reset_permutation(mut self) -> Self {
        self.permutation = (0..self.instances.len()).collect();
        self
    }

    fn split_off(mut self, at: usize) -> [Self; 2] {
        #[allow(clippy::unnecessary_struct_initialization)]
        let metric = Metric { ..self.metric };
        let instances = self.instances.split_off(at);
        let permutation = Vec::new();
        let metadata = self.metadata.split_off(at);
        let right_data = Self {
            metric,
            instances,
            dimensionality_hint: self.dimensionality_hint,
            permutation,
            metadata,
        };
        [self, right_data]
    }
}
