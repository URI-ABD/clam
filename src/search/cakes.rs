use rayon::prelude::*;

use crate::prelude::*;

pub type ClusterResults<'a, T, U> = Vec<(&'a Cluster<'a, T, U>, U)>;

#[derive(Debug, Clone)]
pub struct CAKES<'a, T: Number, U: Number> {
    space: &'a dyn Space<'a, T, U>,
    root: Cluster<'a, T, U>,
    depth: usize,
}

impl<'a, T: Number, U: Number> CAKES<'a, T, U> {
    pub fn new(space: &'a dyn Space<'a, T, U>) -> Self {
        CAKES {
            space,
            root: Cluster::new_root(space).build(),
            depth: 0,
        }
    }

    pub fn build(self, criteria: &crate::PartitionCriteria<T, U>) -> Self {
        let root = self.root.partition(criteria, true);
        let depth = root.max_leaf_depth();
        CAKES {
            space: self.space,
            root,
            depth,
        }
    }

    pub fn space(&self) -> &dyn Space<'a, T, U> {
        self.space
    }

    pub fn data(&self) -> &dyn Dataset<'a, T> {
        self.space.data()
    }

    pub fn metric(&self) -> &dyn Metric<T, U> {
        self.space.metric()
    }

    pub fn root(&self) -> &Cluster<'a, T, U> {
        &self.root
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn radius(&self) -> U {
        self.root.radius()
    }

    pub fn diameter(&self) -> U {
        self.root.radius() * U::from(2).unwrap()
    }

    pub fn batch_rnn_search(&'a self, queries_radii: &[(&[T], U)]) -> Vec<Vec<(usize, U)>> {
        queries_radii
            .par_iter()
            // .iter()
            .map(|(query, radius)| self.rnn_search(query, *radius))
            .collect()
    }

    pub fn rnn_search(&'a self, query: &[T], radius: U) -> Vec<(usize, U)> {
        let [confirmed, straddlers] = self.rnn_tree_search(query, radius);

        let clusters = confirmed
            .into_iter()
            .map(|(c, _)| c)
            .chain(straddlers.into_iter().map(|(c, _)| c))
            .collect::<Vec<_>>();
        self.rnn_leaf_search(query, radius, &clusters)

        // confirmed
        //     .into_iter()
        //     .flat_map(|(c, d)| {
        //         let distances = if c.is_leaf() {
        //             vec![d; c.cardinality()]
        //         } else {
        //             self.space.query_to_many(query, &c.indices())
        //         };
        //         c.indices().into_iter().zip(distances.into_iter())
        //     });

        // straddlers
        //     .into_iter()
        //     .flat_map(|(c, _)| c.indices())
        //     .filter(|&i| self.space.query_to_one(query, i) <= radius)
        //     .chain(confirmed.into_iter().flat_map(|(c, _)| c.indices()))
        //     .collect()
    }

    pub fn rnn_tree_search(&'a self, query: &[T], radius: U) -> [ClusterResults<'a, T, U>; 2] {
        let mut confirmed = Vec::new();
        let mut straddlers = Vec::new();
        let mut candidate_clusters = vec![self.root()];

        while !candidate_clusters.is_empty() {
            let (terminal, non_terminal): (Vec<_>, Vec<_>) = candidate_clusters
                .into_iter()
                .map(|c| (c, self.space.query_to_one(query, c.arg_center())))
                .filter(|&(c, d)| d <= (c.radius() + radius))
                .partition(|&(c, d)| (c.radius() + d) <= radius);
            confirmed.extend(terminal.into_iter());

            let (terminal, non_terminal): (Vec<_>, Vec<_>) = non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
            straddlers.extend(terminal.into_iter());

            candidate_clusters = non_terminal.into_iter().flat_map(|(c, _)| c.children()).collect();
        }

        [confirmed, straddlers]
    }

    pub fn rnn_leaf_search(&self, query: &[T], radius: U, candidate_clusters: &[&Cluster<T, U>]) -> Vec<(usize, U)> {
        self.linear_search(
            query,
            radius,
            Some(candidate_clusters.iter().flat_map(|&c| c.indices()).collect()),
        )
    }

    pub fn batch_knn_search(&'a self, queries: &'a [&[T]], k: usize) -> Vec<Vec<usize>> {
        queries
            .par_iter()
            // .iter()
            .map(|&query| self.knn_search(query, k))
            .collect()
    }

    pub fn knn_search(&'a self, query: &'a [T], k: usize) -> Vec<usize> {
        if k > self.root.cardinality() {
            self.root.indices()
        } else {
            let mut sieve = super::KnnSieve::new(self.root.children().to_vec(), query, k);
            let mut counter = 0;

            while !sieve.is_refined {
                sieve = sieve.refine_step(counter);
                counter += 1;
            }
            sieve.refined_extract()
        }
    }

    fn compute_lfd(&self, distances: &[U], radius: U) -> f64 {
        if radius == U::zero() {
            1.
        } else {
            let half_count = distances
                .iter()
                .filter(|&&d| d <= (radius / U::from(2).unwrap()))
                .count();
            if half_count > 0 {
                ((distances.len() as f64) / (half_count as f64)).log2()
            } else {
                1.
            }
        }
    }

    pub fn batch_knn_by_rnn(&'a self, queries: &[&[T]], k: usize) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            // .iter()
            .map(|&q| self.knn_by_rnn(q, k))
            .collect()
    }

    pub fn knn_by_rnn(&'a self, query: &[T], k: usize) -> Vec<(usize, U)> {
        let mut radius = self.root.radius() / U::from(self.root.cardinality()).unwrap();
        let mut hits = self.rnn_search(query, radius);

        while hits.is_empty() {
            radius = U::from((radius * U::from(2).unwrap()).as_f64() + 1e-12).unwrap();
            hits = self.rnn_search(query, radius);
        }

        while hits.len() < k {
            let distances = hits.iter().map(|(_, d)| *d).collect::<Vec<_>>();
            let lfd = self.compute_lfd(&distances, radius);
            let factor = ((k as f64) / (hits.len() as f64)).powf(1. / (lfd + 1e-12));
            assert!(factor > 1.);
            radius = U::from(radius.as_f64() * factor).unwrap();
            hits = self.rnn_search(query, radius);
        }

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        hits[..k].to_vec()
    }

    pub fn linear_search(&self, query: &[T], radius: U, indices: Option<Vec<usize>>) -> Vec<(usize, U)> {
        let indices = indices.unwrap_or_else(|| self.root.indices());
        let distances = self.space.query_to_many(query, &indices);
        indices
            .into_iter()
            .zip(distances.into_iter())
            .filter(|(_, d)| *d <= radius)
            .collect()
    }

    pub fn batch_linear_search(&self, queries_radii: &[(Vec<T>, U)]) -> Vec<Vec<(usize, U)>> {
        queries_radii
            .par_iter()
            // .iter()
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
        let metric = metric_from_name::<f64, f64>("euclidean", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref(), false);
        let cakes = CAKES::new(&space).build(&crate::PartitionCriteria::new(true));

        let query = &[0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 1.5).into_iter().unzip();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));

        let query = cakes.data().get(1);
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 0.).into_iter().unzip();
        assert_eq!(results.len(), 1);
        assert!(!results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));
    }
}
