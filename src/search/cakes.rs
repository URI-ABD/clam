// use rayon::prelude::*;

use crate::{prelude::*, utils::helpers};

#[derive(Debug)]
pub struct CAKES<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    space: &'a S,
    root: Cluster<'a, T, S>,
    depth: usize,
}

impl<'a, T, S> CAKES<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    pub fn new(space: &'a S) -> Self {
        CAKES {
            space,
            root: Cluster::new_root(space),
            depth: 0,
        }
    }

    pub fn build(mut self, criteria: &crate::PartitionCriteria<'a, T, S>) -> Self {
        self.root = self.root.partition(criteria, true);
        self.depth = self.root.max_leaf_depth();
        self
    }

    pub fn space(&self) -> &S {
        self.space
    }

    pub fn root(&self) -> &Cluster<'a, T, S> {
        &self.root
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn radius(&self) -> f64 {
        self.root.radius()
    }

    pub fn diameter(&self) -> f64 {
        self.root.radius() * 2.
    }

    #[inline(never)]
    pub fn batch_rnn_search(&self, queries_radii: &[(&[T], f64)]) -> Vec<Vec<(usize, f64)>> {
        queries_radii
            // .par_iter()
            .iter()
            .map(|(query, radius)| self.rnn_search(query, *radius))
            .collect()
    }

    pub fn rnn_search(&self, query: &[T], radius: f64) -> Vec<(usize, f64)> {
        let [confirmed, straddlers] = {
            let mut confirmed = Vec::new();
            let mut straddlers = Vec::new();
            let mut candidate_clusters = vec![self.root()];

            while !candidate_clusters.is_empty() {
                let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>) = candidate_clusters
                    .into_iter()
                    .map(|c| (c, c.distance_to_query(query)))
                    .filter(|&(c, d)| d <= (c.radius() + radius))
                    .partition(|&(c, d)| (c.radius() + d) <= radius);
                confirmed.append(&mut terminal);

                let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>) =
                    non_terminal.drain(..).partition(|&(c, _)| c.is_leaf());
                straddlers.append(&mut terminal);

                candidate_clusters = non_terminal.drain(..).flat_map(|(c, _)| c.children()).collect();
            }

            [confirmed, straddlers]
        };

        let hits = confirmed.into_iter().flat_map(|(c, d)| {
            let indices = c.indices();
            let distances = if c.is_leaf() {
                vec![d; indices.len()]
            } else {
                self.space.query_to_many(query, &indices)
            };
            indices.into_iter().zip(distances.into_iter())
        });

        let straddlers = straddlers
            .into_iter()
            .flat_map(|(c, _)| c.indices())
            .collect();
        hits.chain(self.linear_search(query, radius, Some(straddlers)).into_iter())
            .collect()
    }

    // pub fn rnn_leaf_search(
    //     &self,
    //     query: &[T],
    //     radius: f64,
    //     candidate_clusters: &[&Cluster<'a, T, S>],
    // ) -> Vec<(usize, f64)> {
    //     self.linear_search(
    //         query,
    //         radius,
    //         Some(candidate_clusters.iter().flat_map(|&c| c.indices()).collect()),
    //     )
    // }

    // pub fn batch_knn_search(&'a self, queries: &'a [&[T]], k: usize) -> Vec<Vec<usize>> {
    //     queries
    //         .par_iter()
    //         // .iter()
    //         .map(|&query| self.knn_search(query, k))
    //         .collect()
    // }

    // pub fn knn_search(&'a self, query: &'a [T], k: usize) -> Vec<usize> {
    //     if k > self.root.cardinality() {
    //         self.root.indices()
    //     } else {
    //         let mut sieve = super::KnnSieve::new(self.root.children().to_vec(), query, k);
    //         let mut counter = 0;

    //         while !sieve.is_refined {
    //             sieve = sieve.refine_step(counter);
    //             counter += 1;
    //         }
    //         sieve.refined_extract()
    //     }
    // }

    pub fn batch_knn_by_rnn(&'a self, queries: &[&[T]], k: usize) -> Vec<Vec<(usize, f64)>> {
        queries
            // .par_iter()
            .iter()
            .map(|&q| self.knn_by_rnn(q, k))
            .collect()
    }

    pub fn knn_by_rnn(&'a self, query: &[T], k: usize) -> Vec<(usize, f64)> {
        let mut radius = self.root.radius() / self.root.cardinality().as_f64();
        let mut hits = self.rnn_search(query, radius);

        while hits.is_empty() {
            // TODO: Use EPSILON
            radius = radius * 2. + 1e-12;
            hits = self.rnn_search(query, radius);
        }

        while hits.len() < k {
            let distances = hits.iter().map(|(_, d)| *d).collect::<Vec<_>>();
            let lfd = helpers::get_lfd(radius, &distances);
            let factor = ((k as f64) / (hits.len() as f64)).powf(1. / (lfd + 1e-12));
            assert!(factor > 1.);
            radius *= factor;
            hits = self.rnn_search(query, radius);
        }

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        hits[..k].to_vec()
    }

    pub fn linear_search(&self, query: &[T], radius: f64, indices: Option<Vec<usize>>) -> Vec<(usize, f64)> {
        let indices = indices.unwrap_or_else(|| self.root.indices());
        let distances = self.space.query_to_many(query, &indices);
        indices
            .into_iter()
            .zip(distances.into_iter())
            .filter(|(_, d)| *d <= radius)
            .collect()
    }

    pub fn batch_linear_search(&self, queries_radii: &[(Vec<T>, f64)]) -> Vec<Vec<(usize, f64)>> {
        queries_radii
            // .par_iter()
            .iter()
            .map(|(query, radius)| self.linear_search(query, *radius, None))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    use super::CAKES;

    #[test]
    fn test_search() {
        let data = vec![vec![0., 0.], vec![1., 1.], vec![2., 2.], vec![3., 3.]];
        let dataset = crate::Tabular::new(&data, "test_search".to_string());
        let metric = metric_from_name::<f64>("euclidean", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref());
        let cakes = CAKES::new(&space).build(&crate::PartitionCriteria::new(true));

        let query = &[0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 1.5).into_iter().unzip();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));

        let query = cakes.space.data().get(1);
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 0.).into_iter().unzip();
        assert_eq!(results.len(), 1);
        assert!(!results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));
    }
}
