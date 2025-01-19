//! K-NN search using a dataset with search hints.

use core::cmp::Reverse;

use distances::Number;

use crate::{
    cakes::{dataset::HintedDataset, ParHintedDataset, RnnClustered},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Metric, SizedHeap, LFD,
};

use super::{knn_depth_first::d_min, ParSearchAlgorithm, SearchAlgorithm};

/// K-NN search using a dataset with search hints.
pub struct KnnHinted(pub usize);

impl<I, T, C, M, D> SearchAlgorithm<I, T, C, M, D> for KnnHinted
where
    T: Number,
    C: Cluster<T>,
    M: Metric<I, T>,
    D: HintedDataset<I, T, C, M>,
{
    fn name(&self) -> &str {
        "KnnHinted"
    }

    fn radius(&self) -> Option<T> {
        None
    }

    fn k(&self) -> Option<usize> {
        Some(self.0)
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        // The `candidates` heap contains triples of `(d_min, d, c)`.
        let mut candidates = SizedHeap::<(Reverse<T>, T, &C)>::new(None);

        let d = data.query_to_center(metric, query, root);
        let mut hits = vec![(root.arg_center(), d)];
        let mut n_hits = 1;
        candidates.push((Reverse(d_min(root, d)), d, root));

        // While it is possible to have any hit among candidates that is closer
        // than the current closest hit, we keep searching.
        while candidates.peek().is_some_and(|&(Reverse(d), _, _)| d < hits[0].1) {
            let (_, d, c) = candidates.pop().unwrap_or_else(|| unreachable!("We just peeked"));

            if d < hits[0].1 {
                hits.push((c.arg_center(), d));
                hits.swap(0, n_hits);
                n_hits += 1;
            }

            for child in c.children() {
                let d_c = data.query_to_center(metric, query, child);
                candidates.push((Reverse(d_min(child, d_c)), d_c, child));
            }
        }
        let mut logs = vec![format!("hits: {hits:?}")];

        let (i, r) = hits[0];
        logs.push(format!("hints: {:?}", data.hints_for(i)));

        let additive_radius = if data.hints_for(i).contains_key(&self.0) {
            data.hints_for(i)[&self.0]
        } else {
            let hit_distances = hits.iter().map(|&(_, d)| d).collect::<Vec<_>>();
            logs.push(format!("hit_distances: {hit_distances:?}"));
            let (_, max_distance) = crate::utils::arg_max(&hit_distances).unwrap_or((0, T::ZERO));
            logs.push(format!("max_distance: {max_distance:?}"));
            let max_distance = max_distance + T::EPSILON;
            let lfd = LFD::from_radial_distances(&hit_distances, max_distance.half());
            logs.push(format!("lfd: {lfd:?}"));
            let min_multiplier = (1 + hit_distances.len()).as_f32() / self.0.as_f32();
            let multiplier = LFD::multiplier_for_k(lfd, hit_distances.len(), self.0).max(min_multiplier + f32::EPSILON);
            logs.push(format!("multiplier: {multiplier:?}"));
            T::from(max_distance.as_f32() * multiplier)
        };
        logs.push(format!("additive_radius: {additive_radius:?}"));

        let alg = RnnClustered(r + additive_radius);
        let mut hits = alg.search(data, metric, root, query);
        hits.sort_by(|(_, d1), (_, d2)| d1.total_cmp(d2));
        if hits.len() < self.0 {
            for l in logs {
                eprintln!("{l}");
            }
            eprintln!(
                "Expected at least {} hits, got {}, radius: {r}, additive: {additive_radius}",
                self.0,
                hits.len(),
            );
        }
        assert!(hits.len() >= self.0);
        hits.into_iter().take(self.0).collect()
    }
}

impl<I, T, C, M, D> ParSearchAlgorithm<I, T, C, M, D> for KnnHinted
where
    I: Send + Sync,
    T: Number,
    C: ParCluster<T>,
    M: ParMetric<I, T>,
    D: ParHintedDataset<I, T, C, M>,
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        self.search(data, metric, root, query)
    }
}
