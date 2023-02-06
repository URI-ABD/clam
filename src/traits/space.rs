//! Provides the `Space` trait and a struct `TabularSpace` implementing it.

use rand::prelude::*;
// use rayon::prelude::*;

use crate::dataset::Tabular;
use crate::prelude::*;

/// A `Space` represents the combination of a `Dataset` and a `Metric` into a
/// metric space. CLAM is a manifold-mapping framework on such metric spaces.
pub trait Space<'a, T>: std::fmt::Debug + Send + Sync
where
    T: Number + 'a,
{
    /// Returns a reference to the underlying dataset.
    fn data(&self) -> &dyn Dataset<'a, T>;

    /// Returns a reference to the underlying metric.
    fn metric(&self) -> &dyn Metric<T>;

    /// This is built from the names of the dataset and the metric being used.
    fn name(&self) -> String {
        format!("{}__{}", self.data().name(), self.metric().name())
    }

    /// Two instances are considered equal if the distance between them is zero.
    fn are_instances_equal(&self, left: usize, right: usize) -> bool {
        self.one_to_one(left, right) == 0.
    }

    #[inline(never)]
    fn query_to_one(&self, query: &[T], index: usize) -> f64 {
        self.metric().one_to_one(query, self.data().get(index))
    }

    fn query_to_many(&self, query: &[T], indices: &[usize]) -> Vec<f64> {
        // if self.metric().is_expensive() || indices.len() > 1_000 {
        //     indices
        //         .par_iter()
        //         .map(|&index| self.query_to_one(query, index))
        //         .collect()
        // } else {
        //     indices.iter().map(|&index| self.query_to_one(query, index)).collect()
        // }
        indices.iter().map(|&index| self.query_to_one(query, index)).collect()
    }

    /// Computes and returns the distance between two instances.
    fn one_to_one(&self, left: usize, right: usize) -> f64 {
        if left == right {
            0.
        } else {
            self.metric().one_to_one(self.data().get(left), self.data().get(right))
        }
    }

    /// Returns the distances from `left` to each indexed instance in `right`.
    fn one_to_many(&self, left: usize, right: &[usize]) -> Vec<f64> {
        if self.metric().is_expensive() || right.len() > 10_000 {
            right.par_iter().map(|&r| self.one_to_one(left, r)).collect()
            // right.iter().map(|&r| self.one_to_one(left, r)).collect()
        } else {
            right.iter().map(|&r| self.one_to_one(left, r)).collect()
        }
    }

    /// Returns the distances from each indexed instance in `left` to each
    /// indexed instance in `right`.
    fn many_to_many(&self, left: &[usize], right: &[usize]) -> Vec<Vec<f64>> {
        left.iter().map(|&l| self.one_to_many(l, right)).collect()
    }

    /// Returns the all-paris distances between the given indexed instances.
    fn pairwise(&self, indices: &[usize]) -> Vec<Vec<f64>> {
        self.many_to_many(indices, indices)
    }

    /// Chooses `n` unique instances from the given indices and returns their
    /// indices.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of unique instances.
    /// * `indices` - Indices from among which to collect sample.
    fn choose_unique(&self, n: usize, indices: &[usize]) -> Vec<usize> {
        let n = if n < indices.len() { n } else { indices.len() };

        let indices = {
            let mut indices = indices.to_vec();
            indices.shuffle(&mut rand_chacha::ChaCha8Rng::seed_from_u64(42));
            // indices.shuffle(&mut rand::thread_rng());
            indices
        };

        let mut chosen = Vec::new();
        for &i in indices.iter() {
            let is_old = chosen.iter().any(|&o| self.are_instances_equal(i, o));
            if !is_old {
                chosen.push(i);
            }
            if chosen.len() == n {
                break;
            }
        }

        chosen
    }
}

/// A `Space` for a `Tabular` dataset and an arbitrary `Metric`.
pub struct TabularSpace<'a, T: Number> {
    data: &'a Tabular<'a, T>,
    metric: &'a dyn Metric<T>,
}

impl<'a, T: Number> TabularSpace<'a, T> {
    /// # Arguments
    ///
    /// * `data` - Reference to a `Tabular` dataset to use in the metric space.
    /// * `metric` - Distance `Metric` to use with the data.
    /// * `use_cache` - Whether to use a `Cache` for avoid repeated distance
    ///                 computations.
    pub fn new(data: &'a Tabular<T>, metric: &'a dyn Metric<T>) -> TabularSpace<'a, T> {
        TabularSpace { data, metric }
    }
}

impl<'a, T: Number> std::fmt::Debug for TabularSpace<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space")
            .field("data", &self.data.name())
            .field("metric", &self.metric.name())
            .finish()
    }
}

impl<'a, T: Number> Space<'a, T> for TabularSpace<'a, T> {
    fn data(&self) -> &dyn Dataset<'a, T> {
        self.data
    }

    fn metric(&self) -> &dyn Metric<T> {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use crate::dataset::Tabular;
    use crate::prelude::*;

    use super::TabularSpace;

    #[test]
    fn test_space() {
        let data = vec![vec![1., 2., 3.], vec![3., 3., 1.]];
        let dataset = Tabular::new(&data, "test_space".to_string());
        let metric = metric_from_name("euclidean", false).unwrap();
        let space = TabularSpace::new(&dataset, metric.as_ref());

        approx_eq!(f64, space.one_to_one(0, 0), 0.);
        approx_eq!(f64, space.one_to_one(0, 1), 3.);
        approx_eq!(f64, space.one_to_one(1, 0), 3.);
        approx_eq!(f64, space.one_to_one(1, 1), 0.);
    }
}
