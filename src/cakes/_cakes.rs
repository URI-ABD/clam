use distances::Number;
use rayon::prelude::*;

use super::{knn::KnnAlgorithm, rnn::RnnAlgorithm};
use crate::{
    cluster::{PartitionCriteria, Tree},
    dataset::Dataset,
};

#[derive(Debug)]
pub struct CAKES<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> {
    tree: Tree<T, U, D>,
    depth: usize,
}

impl<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> CAKES<T, U, D> {
    pub fn new(data: D, seed: Option<u64>, criteria: PartitionCriteria<T, U>) -> Self {
        let tree = Tree::new(data, seed).partition(&criteria);
        let depth = tree.root().max_leaf_depth();
        Self { tree, depth }
    }

    pub fn tree(&self) -> &Tree<T, U, D> {
        &self.tree
    }

    pub fn data(&self) -> &D {
        self.tree.data()
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn radius(&self) -> U {
        self.tree.radius()
    }

    pub fn batch_rnn_search(&self, queries: &[T], radius: U, algorithm: RnnAlgorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|&query| self.rnn_search(query, radius, algorithm))
            .collect()
    }

    pub fn rnn_search(&self, query: T, radius: U, algorithm: RnnAlgorithm) -> Vec<(usize, U)> {
        algorithm.search(query, radius, &self.tree)
    }

    pub fn batch_knn_search(&self, queries: &[T], k: usize, algorithm: KnnAlgorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|&query| self.knn_search(query, k, algorithm))
            .collect()
    }

    pub fn knn_search(&self, query: T, k: usize, algorithm: KnnAlgorithm) -> Vec<(usize, U)> {
        algorithm.search(query, k, &self.tree)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::core::dataset::VecVec;
    use distances::vectors::euclidean;

    use super::*;

    #[test]
    fn test_search() {
        let data: Vec<&[f32]> = vec![&[0., 0.], &[1., 1.], &[2., 2.], &[3., 3.]];

        let name = "test".to_string();
        let dataset = VecVec::new(data, euclidean, name, false);
        let criteria = PartitionCriteria::new(true);
        let cakes = CAKES::new(dataset, None, criteria);

        let query = vec![0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes
            .rnn_search(&query, 1.5, RnnAlgorithm::Clustered)
            .into_iter()
            .unzip();
        assert_eq!(results.len(), 2);

        let result_points = results.iter().map(|&i| cakes.data().get(i)).collect::<Vec<_>>();
        assert!(result_points.contains(&[0., 0.].as_slice()));
        assert!(result_points.contains(&[1., 1.].as_slice()));

        let query = vec![1., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes
            .rnn_search(&query, 0., RnnAlgorithm::Clustered)
            .into_iter()
            .unzip();
        assert_eq!(results.len(), 1);
        let result_points = results.iter().map(|&i| cakes.data().get(i)).collect::<Vec<_>>();
        assert!(result_points.contains(&[1., 1.].as_slice()));
    }

    #[test]
    fn rnn_search() {
        let data = (-100..=100).map(|x| vec![x as f32]).collect::<Vec<_>>();
        let data = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();
        let data = VecVec::new(data, euclidean, "test".to_string(), false);
        let criteria = PartitionCriteria::new(true);
        let cakes = CAKES::new(data, Some(42), criteria);

        let queries = (-10..=10).step_by(2).map(|x| vec![x as f32]).collect::<Vec<_>>();
        for v in [2, 10, 50] {
            let radius = v as f32;
            let n_hits = 1 + 2 * v;

            for (i, query) in queries.iter().enumerate() {
                let lnn = {
                    let mut hits = cakes.rnn_search(query, radius, RnnAlgorithm::Linear);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
                    hits
                };
                assert_eq!(
                    n_hits,
                    lnn.len(),
                    "Failed linear search: query: {}, radius: {}, linear: {:?}",
                    i,
                    radius,
                    lnn
                );

                let rnn = {
                    let mut hits = cakes.rnn_search(query, radius, RnnAlgorithm::Clustered);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
                    hits
                };
                let lnn_i = lnn.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let rnn_i = rnn.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let diff = lnn_i.difference(&rnn_i).copied().collect::<Vec<_>>();
                assert!(
                    diff.is_empty(),
                    "Failed Clustered search: query: {}, radius: {}\nlnn: {:?}\nrnn: {:?}\ndiff: {:?}",
                    i,
                    radius,
                    lnn,
                    rnn,
                    diff
                );
            }
        }
    }
}
