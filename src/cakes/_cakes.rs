use core::f64::EPSILON;

use distances::Number;
use rayon::prelude::*;

use crate::{
    cluster::{Cluster, PartitionCriteria, Tree},
    dataset::Dataset,
    utils::helpers,
};

#[derive(Debug)]
pub struct CAKES<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> {
    tree: Tree<T, U, D>,
    depth: usize,
}

impl<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> CAKES<T, U, D> {
    pub fn new(data: D, seed: Option<u64>) -> Self {
        Self {
            tree: Tree::new(data, seed),
            depth: 0,
        }
    }

    pub fn build(mut self, criteria: PartitionCriteria<T, U>) -> Self {
        self.tree = self.tree.partition(&criteria);
        self.depth = self.tree.root().max_leaf_depth();
        self
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

    pub fn diameter(&self) -> U {
        self.tree.radius() * U::from(2)
    }

    #[inline(never)]
    pub fn batch_rnn_search(&self, queries: &[T], radius: U) -> Vec<Vec<(usize, U)>> {
        queries.iter().map(|&query| self.rnn_search(query, radius)).collect()
    }

    #[inline(never)]
    pub fn par_batch_rnn_search(&self, queries: &[T], radius: U) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|&query| self.rnn_search(query, radius))
            .collect()
    }

    pub fn rnn_search(&self, query: T, radius: U) -> Vec<(usize, U)> {
        // Tree search.
        let [confirmed, straddlers] = {
            let mut confirmed = Vec::new();
            let mut straddlers = Vec::new();
            let mut candidates = vec![self.tree.root()];

            let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
            while !candidates.is_empty() {
                (terminal, non_terminal) = candidates
                    .drain(..)
                    .map(|c| (c, c.distance_to_instance(self.data(), query)))
                    .filter(|&(c, d)| d <= (c.radius + radius))
                    .partition(|&(c, d)| (c.radius + d) <= radius);
                confirmed.append(&mut terminal);

                (terminal, non_terminal) = non_terminal.drain(..).partition(|&(c, _)| c.is_leaf());
                straddlers.append(&mut terminal);

                candidates = non_terminal
                    .drain(..)
                    .flat_map(|(c, d)| {
                        if d < c.radius {
                            c.overlapping_children(self.data(), query, radius)
                        } else {
                            c.children().unwrap().to_vec()
                        }
                    })
                    .collect();
            }

            [confirmed, straddlers]
        };

        // Leaf Search
        confirmed
            .into_iter()
            .flat_map(|(c, d)| {
                let distances = if c.is_leaf() {
                    vec![d; c.cardinality]
                } else {
                    self.data().query_to_many(query, c.indices(self.data()))
                };
                c.indices(self.data()).iter().copied().zip(distances.into_iter())
            })
            .chain(straddlers.into_iter().flat_map(|(c, _)| {
                let indices = c.indices(self.data());
                let distances = self.data().query_to_many(query, indices);
                indices
                    .iter()
                    .copied()
                    .zip(distances.into_iter())
                    .filter(|&(_, d)| d <= radius)
            }))
            .collect()
    }

    #[inline(never)]
    pub fn batch_knn_search(&self, queries: &[T], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.iter().map(|&query| self.knn_search(query, k)).collect()
    }

    #[inline(never)]
    pub fn par_batch_knn_search(&self, queries: &[T], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|&query| self.knn_search(query, k)).collect()
    }

    pub fn knn_search(&self, query: T, k: usize) -> Vec<(usize, U)> {
        let mut candidates = priority_queue::PriorityQueue::<&Cluster<T, U>, RevNumber<U>>::new();
        let d = self.tree.root().distance_to_instance(self.data(), query);
        candidates.push(self.tree.root(), RevNumber(self.d_min(self.tree.root(), d)));

        let mut hits = priority_queue::PriorityQueue::<usize, OrdNumber<U>>::new();
        // let mut count = 0;

        // stop if we have enough hits and the farthest hit is closer than the closest cluster by delta_min.
        while !(hits.len() >= k
            && (candidates.is_empty() || hits.peek().unwrap().1 .0 < candidates.peek().unwrap().1 .0))
        {
            // println!("count: {count}, candidates: {}, hits: {}", candidates.len(), hits.len());
            self.pop_till_leaf(query, &mut candidates);
            self.leaf_into_hits(query, &mut hits, &mut candidates);
            self.trim_hits(k, &mut hits);
            // count += 1;
        }
        assert!(hits.len() >= k);

        hits.into_iter().map(|(i, OrdNumber(d))| (i, d)).collect()
    }

    #[inline(always)]
    fn d_min(&self, c: &Cluster<T, U>, d: U) -> U {
        if d < c.radius {
            U::zero()
        } else {
            d - c.radius
        }
    }

    // pop from the top of `candidates` until the top candiadte is a leaf cluster.
    fn pop_till_leaf(&self, query: T, candidates: &mut priority_queue::PriorityQueue<&Cluster<T, U>, RevNumber<U>>) {
        while !candidates.peek().unwrap().0.is_leaf() {
            let [l, r] = candidates.pop().unwrap().0.children().unwrap();
            let [dl, dr] = [
                l.distance_to_instance(self.data(), query),
                r.distance_to_instance(self.data(), query),
            ];
            candidates.push(l, RevNumber(self.d_min(l, dl)));
            candidates.push(r, RevNumber(self.d_min(r, dr)));
        }
    }

    // pop a single leaf from the top of candidates and add those points to hits.
    fn leaf_into_hits(
        &self,
        query: T,
        hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>,
        candidates: &mut priority_queue::PriorityQueue<&Cluster<T, U>, RevNumber<U>>,
    ) {
        let (leaf, RevNumber(d)) = candidates.pop().unwrap();
        let is = leaf.indices(self.data());
        let ds = if leaf.is_singleton() {
            vec![d; is.len()]
        } else {
            self.data().query_to_many(query, is)
        };
        is.iter().zip(ds.into_iter()).for_each(|(&i, d)| {
            hits.push(i, OrdNumber(d));
        });
    }

    // reduce hits down to k elements, including ties for the kth farthest element.
    fn trim_hits(&self, k: usize, hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>) {
        if hits.len() > k {
            let mut potential_ties = vec![hits.pop().unwrap()];
            while hits.len() >= k {
                let item = hits.pop().unwrap();
                if item.1 .0 < potential_ties.last().unwrap().1 .0 {
                    potential_ties.clear();
                }
                potential_ties.push(item);
            }
            hits.extend(potential_ties.drain(..));
        }
    }

    // pub fn knn_search(&self, query: &T, k: usize) -> Vec<(usize, U)> {
    //     let mut sieve = KnnSieve::new(&self.root, query, k);
    //     while !sieve.is_refined() {
    //         sieve.refine_step();
    //     }
    //     sieve.extract()
    // }

    #[inline(never)]
    pub fn batch_knn_by_rnn(&self, queries: &[T], k: usize) -> Vec<Vec<(usize, U)>> {
        queries
            // .par_iter()
            .iter()
            .map(|&query| self.knn_by_rnn(query, k))
            .collect()
    }

    pub fn knn_by_rnn(&self, query: T, k: usize) -> Vec<(usize, U)> {
        let mut radius = EPSILON + self.tree.root().radius.as_f64() / self.tree.root().cardinality.as_f64();
        let mut hits = self.rnn_search(query, U::from(radius));

        while hits.is_empty() {
            radius = EPSILON + 2. * radius;
            hits = self.rnn_search(query, U::from(radius));
        }

        while hits.len() < k {
            let distances = hits.iter().map(|(_, d)| *d).collect::<Vec<_>>();
            let lfd = helpers::compute_lfd(U::from(radius), &distances);
            let factor = (k.as_f64() / hits.len().as_f64()).powf(1. / (lfd + EPSILON));
            assert!(factor > 1.);
            radius *= factor;
            hits = self.rnn_search(query, U::from(radius));
        }

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        hits[..k].to_vec()
    }

    // TODO: Add knn version
    #[inline(never)]
    pub fn batch_linear_search(&self, queries: &[T], radius: U) -> Vec<Vec<(usize, U)>> {
        queries
            // .par_iter()
            .iter()
            .map(|&query| self.linear_search(query, radius, None))
            .collect()
    }

    // TODO: Add knn version
    pub fn linear_search(&self, query: T, radius: U, indices: Option<&[usize]>) -> Vec<(usize, U)> {
        let indices = indices.unwrap_or_else(|| self.tree.root().indices(self.data()));
        let distances = self.data().query_to_many(query, indices);
        indices
            .iter()
            .copied()
            .zip(distances.into_iter())
            .filter(|(_, d)| *d <= radius)
            .collect()
    }
}

struct OrdNumber<T: Number>(T);

impl<T: Number> PartialEq for OrdNumber<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Number> Eq for OrdNumber<T> {}

impl<T: Number> PartialOrd for OrdNumber<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: Number> Ord for OrdNumber<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

struct RevNumber<T: Number>(T);

impl<T: Number> PartialEq for RevNumber<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Number> Eq for RevNumber<T> {}

impl<T: Number> PartialOrd for RevNumber<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl<T: Number> Ord for RevNumber<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.partial_cmp(&self.0).unwrap()
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
        let cakes = CAKES::new(dataset, None).build(criteria);

        let query = vec![0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(&query, 1.5).into_iter().unzip();
        assert_eq!(results.len(), 2);

        let result_points = results.iter().map(|&i| cakes.data().get(i)).collect::<Vec<_>>();
        assert!(result_points.contains(&[0., 0.].as_slice()));
        assert!(result_points.contains(&[1., 1.].as_slice()));

        let query = vec![1., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(&query, 0.).into_iter().unzip();
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
        let cakes = CAKES::new(data, Some(42)).build(criteria);

        let queries = (-10..=10).step_by(2).map(|x| vec![x as f32]).collect::<Vec<_>>();
        for v in [2, 10, 50] {
            let radius = v as f32;
            let n_hits = 1 + 2 * v;

            for (i, query) in queries.iter().enumerate() {
                let lnn = {
                    let mut hits = cakes.linear_search(query, radius, None);
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
                    let mut hits = cakes.rnn_search(query, radius);
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
