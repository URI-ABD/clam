use rayon::prelude::*;

use crate::prelude::*;

pub type SearchHistory<'a, T, U> = Vec<(&'a Cluster<'a, T, U>, U)>;

#[derive(Debug, Clone)]
pub struct CAKES<'a, T: Number, U: Number> {
    space: &'a dyn Space<T, U>,
    root: Cluster<'a, T, U>,
    depth: usize,
}

impl<'a, T: Number, U: Number> CAKES<'a, T, U> {
    pub fn new(space: &'a dyn Space<T, U>) -> Self {
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

    pub fn space(&self) -> &dyn Space<T, U> {
        self.space
    }

    pub fn data(&self) -> &dyn Dataset<T> {
        self.space.data()
    }

    pub fn metric(&self) -> &dyn Metric<T, U> {
        self.space.metric()
    }

    pub fn root(&self) -> &Cluster<T, U> {
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

    pub fn batch_rnn_search(&self, queries_radii: &[(Vec<T>, U)]) -> Vec<Vec<(usize, U)>> {
        queries_radii
            .par_iter()
            .map(|(query, radius)| self.rnn_search(query, *radius))
            .collect()
    }

    pub fn rnn_search(&self, query: &[T], radius: U) -> Vec<(usize, U)> {
        let candidate_clusters = self.rnn_tree_search(query, radius).1;

        if candidate_clusters.is_empty() {
            Vec::new()
        } else {
            self.rnn_leaf_search(query, radius, &candidate_clusters)
        }
    }

    pub fn rnn_tree_search(&self, query: &[T], radius: U) -> (SearchHistory<T, U>, Vec<&Cluster<T, U>>) {
        let mut history = Vec::new();
        let mut hits = Vec::new();
        let mut candidate_clusters = vec![self.root()];

        while !candidate_clusters.is_empty() {
            let centers: Vec<_> = candidate_clusters.iter().map(|c| c.center()).collect();
            let distances = if self.metric().is_expensive() || centers.len() > 1000 {
                self.metric().par_one_to_many(query, &centers)
            } else {
                self.metric().one_to_many(query, &centers)
            };
            let close_enough: Vec<_> = candidate_clusters
                .into_iter()
                .zip(distances.into_iter())
                .filter(|(c, d)| *d <= (c.radius() + radius))
                .collect();
            history.extend(close_enough.iter().cloned());
            let (terminal, non_terminal): (Vec<_>, Vec<_>) = close_enough
                .into_iter()
                .partition(|(c, d)| c.is_leaf() || (c.radius() + *d) <= radius);
            hits.extend(terminal.into_iter().map(|(c, _)| c));
            candidate_clusters = non_terminal
                .into_iter()
                .flat_map(|(c, _)| [c.left_child(), c.right_child()])
                .collect();
        }

        (history, hits)
    }

    pub fn rnn_leaf_search(&self, query: &[T], radius: U, candidate_clusters: &[&Cluster<T, U>]) -> Vec<(usize, U)> {
        self.linear_search(
            query,
            radius,
            Some(candidate_clusters.iter().flat_map(|c| c.indices()).collect()),
        )
    }

    pub fn batch_knn_search(&self, queries_ks: &[(Vec<T>, usize)]) -> Vec<Vec<(usize, U)>> {
        queries_ks
            .par_iter()
            .map(|(query, k)| self.knn_search(query, *k))
            .collect()
    }

    pub fn knn_search(&self, query: &[T], k: usize) -> Vec<(usize, U)> {
        let candidate_clusters = self.knn_tree_search(query, k);
        assert!(candidate_clusters.len() >= k);

        let knn = self.knn_leaf_search(query, k, &candidate_clusters);
        assert!(knn.len() == k);

        knn
    }

    pub fn knn_tree_search(&self, query: &'a [T], k: usize) -> Vec<(&Cluster<T, U>, U)> {
        let mut sieve = super::KnnSieve::new(vec![&self.root], query, k);

        while !sieve.are_all_leaves() {
            sieve = sieve.replace_with_child_clusters().filter();
        }

        sieve.clusters.into_iter().zip(sieve.deltas_0.into_iter()).collect()
    }

    #[allow(unused_variables)]
    pub fn knn_leaf_search(&self, query: &[T], k: usize, candidate_clusters: &[(&Cluster<T, U>, U)]) -> Vec<(usize, U)> {
        todo!()
    }

    pub fn linear_search(&self, query: &[T], radius: U, indices: Option<Vec<usize>>) -> Vec<(usize, U)> {
        let indices = indices.unwrap_or_else(|| self.root.indices());

        if self.metric().is_expensive() || indices.len() > 1000 {
            indices
                .into_par_iter()
                .map(|i| (i, self.data().get(i)))
                .map(|(i, y)| (i, self.metric().one_to_one(query, &y)))
                .filter(|(_, d)| *d <= radius)
                .collect()
        } else {
            indices
                .into_iter()
                .map(|i| (i, self.data().get(i)))
                .map(|(i, y)| (i, self.metric().one_to_one(query, &y)))
                .filter(|(_, d)| *d <= radius)
                .collect()
        }
    }

    pub fn batch_linear_search(&self, queries_radii: &[(Vec<T>, U)]) -> Vec<Vec<(usize, U)>> {
        queries_radii
            .par_iter()
            .map(|(query, radius)| self.linear_search(query, *radius, None))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::prelude::*;

    use super::CAKES;

    #[test]
    fn test_search() {
        let data = vec![vec![0., 0.], vec![1., 1.], vec![2., 2.], vec![3., 3.]];
        let dataset = crate::Tabular::new(&data, "test_search".to_string());
        let metric = metric_from_name("euclidean", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref(), false);
        let cakes = CAKES::new(&space).build(&crate::PartitionCriteria::new(true));

        let query = &[0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 1.5).into_iter().unzip();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));

        let query = Arc::new(cakes.data()).get(1);
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(&query, 0.).into_iter().unzip();
        assert_eq!(results.len(), 1);
        assert!(!results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));
    }
}