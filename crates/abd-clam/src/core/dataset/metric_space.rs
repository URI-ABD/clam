//! `MetricSpace` is a trait for datasets that have a distance function.

use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

/// `MetricSpace` is a trait for datasets that have a distance function.
///
/// # Type Parameters
///
/// * `I`: The type of the instances, i.e. each data point in the dataset.
/// * `U`: The type of the distance values.
pub trait MetricSpace<I, U: Number> {
    /// Whether the distance function provides an identity.
    ///
    /// Identity is defined as: `d(x, y) = 0 <=> x = y`.
    fn identity(&self) -> bool;

    /// Whether the distance function is non-negative.
    ///
    /// Non-negativity is defined as: `d(x, y) >= 0`.
    fn non_negativity(&self) -> bool;

    /// Whether the distance function is symmetric.
    ///
    /// Symmetry is defined as: `d(x, y) = d(y, x) for all x, y`.
    fn symmetry(&self) -> bool;

    /// Whether the distance function satisfies the triangle inequality.
    ///
    /// The triangle inequality is defined as: `d(x, y) + d(y, z) >= d(x, z) for all x, y, z`.
    fn triangle_inequality(&self) -> bool;

    /// Whether the distance function is expensive to compute.
    ///
    /// A distance function is expensive if its asymptotic complexity is greater
    /// than O(n) where n is the dimensionality of the dataset.
    fn expensive(&self) -> bool;

    /// Returns the distance function
    fn distance_function(&self) -> fn(&I, &I) -> U;

    /// Calculates the distance between two instances.
    fn one_to_one(&self, a: &I, b: &I) -> U {
        self.distance_function()(a, b)
    }

    /// Whether two instances are equal.
    ///
    /// This will always return `false` when the distance function does not
    /// provide an identity.
    fn equal(&self, a: &I, b: &I) -> bool {
        self.identity() && (self.one_to_one(a, b) == U::ZERO)
    }

    /// Calculates the distances between an instance and a collection of instances.
    fn one_to_many<A: Copy>(&self, a: &I, b: &[(A, &I)]) -> Vec<(A, U)> {
        b.iter().map(|&(i, x)| (i, self.one_to_one(a, x))).collect()
    }

    /// Calculates the distances between two collections of instances.
    fn many_to_many<A: Copy, B: Copy>(&self, a: &[(A, &I)], b: &[(B, &I)]) -> Vec<Vec<(A, B, U)>> {
        a.iter()
            .map(|&(i, x)| b.iter().map(move |&(j, y)| (i, j, self.one_to_one(x, y))).collect())
            .collect()
    }

    /// Calculates the distances between all given pairs of instances.
    fn pairs<A: Copy, B: Copy>(&self, pairs: &[(A, &I, B, &I)]) -> Vec<(A, B, U)> {
        pairs
            .iter()
            .map(|&(i, a, j, b)| (i, j, self.one_to_one(a, b)))
            .collect()
    }

    /// Calculates all pairwise distances between instances.
    fn pairwise<A: Copy>(&self, a: &[(A, &I)]) -> Vec<Vec<(A, A, U)>> {
        if self.symmetry() {
            let mut matrix = a
                .iter()
                .map(|&(i, _)| a.iter().map(move |&(j, _)| (i, j, U::ZERO)).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            for (i, &(p, x)) in a.iter().enumerate() {
                let pairs = a.iter().skip(i + 1).map(|&(q, y)| (p, x, q, y)).collect::<Vec<_>>();
                let distances = self.pairs(&pairs);
                distances
                    .into_iter()
                    .enumerate()
                    .map(|(j, d)| (j + i + 1, d))
                    .for_each(|(j, (p, q, d))| {
                        matrix[i][j] = (p, q, d);
                        matrix[j][i] = (q, p, d);
                    });
            }

            if !self.identity() {
                // compute the diagonal for non-metrics
                let pairs = a.iter().map(|&(p, x)| (p, x, p, x)).collect::<Vec<_>>();
                let distances = self.pairs(&pairs);
                distances
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, (p, q, d))| matrix[i][i] = (p, q, d));
            }

            matrix
        } else {
            self.many_to_many(a, a)
        }
    }

    /// Chooses a subset of instances that are unique with respect to the
    /// distance function.
    ///
    /// If the distance function does not provide an identity, this will return
    /// a random subset of the instances.
    ///
    /// # Arguments
    ///
    /// * `instances` - A collection of instances.
    /// * `choose` - The number of instances to choose.
    /// * `seed` - A seed for the random number generator.
    ///
    /// # Returns
    ///
    /// The indices of the chosen instances in `instances`.
    fn choose_unique<'a, A: Copy>(
        &'a self,
        instances: &[(A, &'a I)],
        choose: usize,
        seed: Option<u64>,
    ) -> Vec<(A, &'a I)> {
        let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

        if self.identity() {
            let mut choices = Vec::with_capacity(choose);
            for &(i, a) in instances {
                if !choices.iter().any(|&(_, b)| self.equal(a, b)) {
                    choices.push((i, a));
                }

                if choices.len() == choose {
                    break;
                }
            }
            choices
        } else {
            let mut instances = instances.to_vec();
            instances.shuffle(&mut rng);
            instances.truncate(choose);
            instances
        }
    }

    /// Calculates the geometric median of a collection of instances.
    ///
    /// The geometric median is the instance that minimizes the sum of distances
    /// to all other instances.
    ///
    /// # Arguments
    ///
    /// * `instances` - A collection of instances.
    ///
    /// # Returns
    ///
    /// The index of the geometric median in `instances`.
    fn median<'a, A: Copy>(&'a self, instances: &[(A, &'a I)]) -> (A, &'a I) {
        let distances = self.pairwise(instances);
        let mut median = 0;
        let mut min_sum = U::MAX;

        for (i, row) in distances.into_iter().enumerate() {
            let sum = row.into_iter().map(|(_, _, d)| d).sum();
            if sum < min_sum {
                min_sum = sum;
                median = i;
            }
        }

        instances[median]
    }
}

/// An extension of `MetricSpace` that provides parallel implementations of
/// distance calculations.
#[allow(clippy::module_name_repetitions)]
pub trait ParMetricSpace<I: Send + Sync, U: Number>: MetricSpace<I, U> + Send + Sync {
    /// Calculates the distances between an instance and a collection of
    /// instances, in parallel.
    fn par_one_to_many<A: Copy + Send + Sync>(&self, a: &I, b: &[(A, &I)]) -> Vec<(A, U)> {
        b.par_iter().map(|&(i, x)| (i, self.one_to_one(a, x))).collect()
    }

    /// Calculates the distances between two collections of instances, in parallel.
    fn par_many_to_many<A: Copy + Send + Sync, B: Copy + Send + Sync>(
        &self,
        a: &[(A, &I)],
        b: &[(B, &I)],
    ) -> Vec<Vec<(A, B, U)>> {
        a.par_iter()
            .map(|&(i, x)| b.par_iter().map(move |&(j, y)| (i, j, self.one_to_one(x, y))).collect())
            .collect()
    }

    /// Calculates the distances between all given pairs of instances, in parallel.
    #[allow(clippy::type_complexity)]
    fn par_pairs<A: Copy + Send + Sync, B: Copy + Send + Sync>(&self, pairs: &[(A, &I, B, &I)]) -> Vec<(A, B, U)> {
        pairs
            .par_iter()
            .map(|&(i, a, j, b)| (i, j, self.one_to_one(a, b)))
            .collect()
    }

    /// Calculates all pairwise distances between instances, in parallel.
    fn par_pairwise<A: Copy + Send + Sync>(&self, a: &[(A, &I)]) -> Vec<Vec<(A, A, U)>> {
        if self.symmetry() {
            let mut matrix = a
                .iter()
                .map(|&(i, _)| a.iter().map(move |&(j, _)| (i, j, U::ZERO)).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            for (i, &(p, x)) in a.iter().enumerate() {
                let pairs = a
                    .iter()
                    .skip(i + 1)
                    .map(move |&(q, y)| (p, x, q, y))
                    .collect::<Vec<_>>();
                let distances = self.par_pairs(&pairs);
                distances
                    .into_iter()
                    .enumerate()
                    .map(|(j, d)| (j + i + 1, d))
                    .for_each(|(j, (a, b, d))| {
                        matrix[i][j] = (a, b, d);
                        matrix[j][i] = (a, b, d);
                    });
            }

            if !self.identity() {
                // compute the diagonal for non-metrics
                let pairs = a.iter().map(|&(i, x)| (i, x, i, x)).collect::<Vec<_>>();
                let distances = self.par_pairs(&pairs);
                distances
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, (a, b, d))| matrix[i][i] = (a, b, d));
            }

            matrix
        } else {
            self.par_many_to_many(a, a)
        }
    }

    /// Chooses a subset of instances that are unique with respect to the
    /// distance function.
    ///
    /// If the distance function does not provide an identity, this will return
    /// a random subset of the instances.
    ///
    /// # Arguments
    ///
    /// * `instances` - A collection of instances.
    /// * `choose` - The number of instances to choose.
    /// * `seed` - A seed for the random number generator.
    ///
    /// # Returns
    ///
    /// The indices of the chosen instances in `instances`.
    fn par_choose_unique<'a, A: Copy + Send + Sync>(
        &'a self,
        instances: &[(A, &'a I)],
        choose: usize,
        seed: Option<u64>,
    ) -> Vec<(A, &'a I)> {
        let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

        if self.identity() {
            let mut choices = Vec::with_capacity(choose);
            for &(i, a) in instances {
                if !choices.par_iter().any(|&(_, b)| self.equal(a, b)) {
                    choices.push((i, a));
                }

                if choices.len() == choose {
                    break;
                }
            }
            choices
        } else {
            let mut instances = instances.to_vec();
            instances.shuffle(&mut rng);
            instances.truncate(choose);
            instances
        }
    }

    /// Calculates the geometric median of a collection of instances.
    ///
    /// The geometric median is the instance that minimizes the sum of distances
    /// to all other instances.
    ///
    /// # Arguments
    ///
    /// * `instances` - A collection of instances.
    ///
    /// # Returns
    ///
    /// The index of the geometric median in `instances`.
    fn par_median<'a, A: Copy + Send + Sync>(&self, instances: &[(A, &'a I)]) -> (A, &'a I) {
        let distances = self.par_pairwise(instances);
        let mut median = 0;
        let mut min_sum = U::MAX;

        for (i, row) in distances.into_iter().enumerate() {
            let sum = row.into_iter().map(|(_, _, d)| d).sum();
            if sum < min_sum {
                min_sum = sum;
                median = i;
            }
        }

        instances[median]
    }
}
