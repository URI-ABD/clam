//! Extension of the `Tree` struct with methods for performing multiple sequence alignment using the MUSALS algorithm.

use std::collections::HashMap;

use crate::{DistanceValue, Tree};

use super::{AlignedSequence, CostMatrix};

mod par_tree;

impl<Id, T, A, M> Tree<Id, AlignedSequence, T, A, M>
where
    T: DistanceValue,
    M: Fn(&String, &String) -> T,
{
    /// Aligns the sequences in the tree using `MuSAlS`.
    pub(crate) fn align_bottom_up(&mut self, cost_matrix: &CostMatrix<T>) {
        let (leaves, mut parents_in_waiting) = self
            .items
            .iter()
            .filter_map(|(_, _, loc)| {
                loc.as_cluster()
                    .map(|c| c.child_center_indices().map_or((c.center_index, 0), |cids| (c.center_index, cids.len())))
            })
            .partition::<HashMap<_, _>, _>(|&(_, n)| n == 0);

        let mut frontier = leaves
            .into_keys()
            .inspect(|&id| {
                self.align_leaf(id, cost_matrix);
                if let Some(waiting) = self.items[id]
                    .2
                    .as_cluster()
                    .and_then(|c| c.parent_center_index)
                    .and_then(|pid| parents_in_waiting.get_mut(&pid))
                {
                    *waiting -= 1;
                }
            })
            .collect::<Vec<_>>();

        let mut full_parents: HashMap<_, _>;
        while !frontier.is_empty() {
            (full_parents, parents_in_waiting) = parents_in_waiting.into_iter().partition(|&(_, waiting)| waiting == 0);

            frontier = full_parents
                .into_keys()
                .inspect(|&id| {
                    self.align_parent(id, cost_matrix);
                    if let Some(waiting) = self.items[id]
                        .2
                        .as_cluster()
                        .and_then(|c| c.parent_center_index)
                        .and_then(|pid| parents_in_waiting.get_mut(&pid))
                    {
                        *waiting -= 1;
                    }
                })
                .collect::<Vec<_>>();
        }
    }

    /// Aligns the sequences in a leaf cluster.
    fn align_leaf(&mut self, id: usize, cost_matrix: &CostMatrix<T>) {
        let car = self.items[id]
            .2
            .as_cluster()
            .map_or_else(|| unreachable!("The caller ensures that the id is a valid leaf cluster."), |c| c.cardinality);

        for i in 1..car {
            let center = &self.items[id].1;
            let target = &self.items[id + i].1;
            let [c_gaps, t_gaps] = center.compute_gap_indices(target, cost_matrix);
            self.insert_gaps(&c_gaps, id..id + i);
            self.items[id + i].1.insert_gaps(&t_gaps);
        }
    }

    /// Aligns the sequences in a parent cluster by merging the alignments of its child clusters.
    fn align_parent(&mut self, id: usize, cost_matrix: &CostMatrix<T>) {
        let cids = self.items[id]
            .2
            .as_cluster()
            .and_then(|c| c.child_center_indices().map(<[_]>::to_vec))
            .unwrap_or_else(|| unreachable!("The caller ensures that the id is a valid parent cluster."));

        // Align the child clusters together
        let l_id = cids[0];
        let mut aligned_car = self.items[l_id]
            .2
            .as_cluster()
            .map_or_else(|| unreachable!("The tree ensures that every child id is a valid cluster."), |c| c.cardinality);
        for &r_id in &cids[1..] {
            let left = &self.items[l_id].1;
            let right = &self.items[r_id].1;
            let [l_gaps, r_gaps] = left.compute_gap_indices(right, cost_matrix);

            self.insert_gaps(&l_gaps, l_id..l_id + aligned_car);

            let r_car = self.items[r_id]
                .2
                .as_cluster()
                .map_or_else(|| unreachable!("The tree ensures that every child id is a valid cluster."), |c| c.cardinality);

            self.insert_gaps(&r_gaps, r_id..r_id + r_car);
            aligned_car += r_car;
        }

        // Align the parent cluster to the closest child cluster
        let closest_cid = {
            let parent_seq = self.items[id].1.original();
            cids.iter()
                .map(|&cid| (cid, (self.metric)(&parent_seq, &self.items[cid].1.original())))
                .min_by_key(|(_, d)| crate::utils::MinItem((), *d))
                .map_or_else(|| unreachable!("Every parent cluster should have at least two child clusters."), |(cid, _)| cid)
        };

        let left = &self.items[id].1;
        let right = &self.items[closest_cid].1;
        let [l_gaps, r_gaps] = left.compute_gap_indices(right, cost_matrix);

        self.insert_gaps(&l_gaps, id..id + 1);
        self.insert_gaps(&r_gaps, (id + 1)..(id + 1 + aligned_car));
    }

    /// Inserts gaps at the given indices to each of the sequences in the given range of indices.
    fn insert_gaps(&mut self, gaps: &[usize], seq_range: core::ops::Range<usize>) {
        for (_, seq, _) in &mut self.items[seq_range] {
            seq.insert_gaps(gaps);
        }
    }
}
