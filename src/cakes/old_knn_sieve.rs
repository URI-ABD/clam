use std::cmp::Ordering;

use crate::core::cluster::Cluster;
use crate::core::dataset::Dataset;
use crate::core::number::Number;
use crate::search::find_kth;

#[derive(Debug, Clone)]
pub enum Delta {
    D,
    Max,
    Min,
}

#[derive(Debug, Clone)]
pub struct Grain<'a, T: Number, U: Number> {
    pub c: &'a Cluster<'a, T, U>,
    pub d: U,
    pub d_min: U,
    pub d_max: U,
}

impl<'a, T: Number, U: Number> std::fmt::Display for Grain<'a, T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "name: {}, d {:6.12}, d_min: {:6.12}, d_max: {:6.12}",
            self.c.name_str(),
            self.d,
            self.d_min,
            self.d_max
        )
    }
}

impl<'a, T: Number, U: Number> PartialEq for Grain<'a, T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.d_max == other.d_max
    }
}

impl<'a, T: Number, U: Number> Eq for Grain<'a, T, U> {}

impl<'a, T: Number, U: Number> PartialOrd for Grain<'a, T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.d_max.partial_cmp(&other.d_max)
    }
}

impl<'a, T: Number, U: Number> Ord for Grain<'a, T, U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a, T: Number, U: Number> Grain<'a, T, U> {
    pub fn new(c: &'a Cluster<'a, T, U>, d: U) -> Self {
        let d_min = if d > c.radius() { d - c.radius() } else { U::zero() };
        let d_max = d + c.radius();
        Self { c, d, d_min, d_max }
    }

    pub fn is_inside(&self, threshold: U) -> bool {
        self.d_max <= threshold
    }

    pub fn is_outside(&self, threshold: U) -> bool {
        self.d_min > threshold
    }

    pub fn is_straddling(&self, threshold: U) -> bool {
        !(self.is_inside(threshold) || self.is_outside(threshold))
    }

    pub fn ord_by_d(&self, other: &Self) -> Ordering {
        self.d.partial_cmp(&other.d).unwrap()
    }

    pub fn ord_by_d_min(&self, other: &Self) -> Ordering {
        self.d_min.partial_cmp(&other.d_min).unwrap()
    }

    pub fn ord_by_d_max(&self, other: &Self) -> Ordering {
        self.d_max.partial_cmp(&other.d_max).unwrap()
    }
}

#[derive(Debug)]
pub struct OrdNumber<U: Number> {
    pub number: U,
}

impl<U: Number> PartialEq for OrdNumber<U> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl<U: Number> Eq for OrdNumber<U> {}

impl<U: Number> PartialOrd for OrdNumber<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl<U: Number> Ord for OrdNumber<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Struct for facilitating knn tree search
/// `clusters` represents the list of candidate Clusters
/// The ith element of `cumulative_cardinalities` represents the sum of cardinalities of the 0th through ith Cluster in
/// `clusters`.
#[derive(Debug)]
pub struct KnnSieve<'a, T: Number, U: Number> {
    dataset: &'a dyn Dataset<'a, T, U>,
    pub grains: Vec<Grain<'a, T, U>>,
    query: &'a [T],
    pub k: usize,
    pub cumulative_cardinalities: Vec<usize>,
    pub is_refined: bool,
    insiders: Vec<Grain<'a, T, U>>,
    straddlers: Vec<Grain<'a, T, U>>,
    pub guaranteed_cardinalities: Vec<usize>,
    threshold: U,
    pub hits: priority_queue::DoublePriorityQueue<usize, OrdNumber<U>>,
}

impl<'a, T: Number, U: Number> std::fmt::Display for KnnSieve<'a, T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "k: {}, \n\n{}.\n",
            self.k,
            self.grains
                .iter()
                .zip(self.cumulative_cardinalities.iter())
                .enumerate()
                .map(|(i, (g, c))| format!("i: {}, car: {}, grain: {}", i, c, g))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}

fn grain_filter<T: Number, U: Number>(mut grains: Vec<Grain<T, U>>, threshold: U) -> [Vec<Grain<T, U>>; 2] {
    let (insiders, straddlers) = grains
        .drain(..)
        .filter(|g| !g.is_outside(threshold))
        .partition(|g| g.is_inside(threshold));

    [insiders, straddlers]
}

impl<'a, T: Number, U: Number> KnnSieve<'a, T, U> {
    pub fn new(clusters: Vec<&'a Cluster<'a, T, U>>, query: &'a [T], k: usize) -> Self {
        let dataset = clusters.first().unwrap().dataset();
        let grains = clusters
            .iter()
            .map(|&c| Grain::new(c, dataset.query_to_one(query, c.arg_center())))
            .collect();
        let cumulative_cardinalities = clusters
            .iter()
            .scan(0, |acc, c| {
                *acc += c.cardinality();
                Some(*acc)
            })
            .collect();
        Self {
            dataset,
            grains,
            query,
            k,
            cumulative_cardinalities,
            is_refined: false,
            insiders: vec![],
            straddlers: vec![],
            guaranteed_cardinalities: vec![],
            threshold: U::zero(),
            hits: priority_queue::DoublePriorityQueue::new(),
        }
    }

    #[inline(never)]
    pub fn refine_step(mut self) -> Self {
        // log::debug!("");
        // log::debug!("Step {}: Starting with {} grains ...", step, self.grains.len());

        let mut thresholds = self
            .grains
            .iter()
            .map(|g| (g.d, if g.c.is_leaf() { g.c.cardinality() } else { 1 }))
            .chain(
                self.grains
                    .iter()
                    .filter(|g| !g.c.is_leaf())
                    .map(|g| (g.d_max, g.c.cardinality() - 1)),
            )
            .chain(self.hits.iter().map(|(_, d)| (d.number, 1)))
            .collect::<Vec<_>>();
        let (index, (threshold, _)) = find_kth::find_kth_threshold(&mut thresholds, self.k);

        if index > 0 {
            (0..index).for_each(|i| {
                assert!(
                    thresholds[i].0 <= threshold,
                    "Failed in smaller partition at index: {}, i: {}, new_threshold: {} vs old_threshold: {}",
                    index,
                    i,
                    thresholds[i].0,
                    threshold
                )
            });
        }
        ((index + 1)..thresholds.len()).for_each(|i| {
            assert!(
                thresholds[i].0 > threshold,
                "Failed in larger partition at index: {}, i: {}, new_threshold: {} vs old_threshold: {}",
                index,
                i,
                thresholds[i].0,
                threshold
            )
        });

        let num_guaranteed = thresholds[..=index].iter().fold(0, |acc, &(_, c)| acc + c);
        assert!(
            num_guaranteed >= self.k,
            "Too few guarantees {} vs {}, index: {}, threshold: {}",
            num_guaranteed,
            self.k,
            index,
            threshold,
        );
        // log::debug!(
        //     "Step {}: Chose index {}, threshold {}, and guaranteed {} points, with {} in hits",
        //     step,
        //     index,
        //     threshold,
        //     num_guaranteed,
        //     self.hits.len(),
        // );

        while !self.hits.is_empty() && self.hits.peek_max().unwrap().1.number > threshold {
            self.hits.pop_max().unwrap();
        }

        (self.insiders, self.straddlers) = self
            .grains
            .drain(..)
            .filter(|g| !g.is_outside(threshold))
            .partition(|g| g.is_inside(threshold));
        // log::debug!(
        //     "Step {}: Got {} insiders and {} straddlers, with {} in hits ...",
        //     step,
        //     self.insiders.len(),
        //     self.straddlers.len(),
        //     self.hits.len(),
        // );

        let (small_insiders, insiders): (Vec<_>, Vec<_>) = self
            .insiders
            .drain(..)
            .partition(|g| (g.c.cardinality() <= self.k) || g.c.is_leaf());
        self.insiders = insiders;
        small_insiders.into_iter().for_each(|g| {
            let new_hits =
                g.c.indices()
                    // .into_par_iter()
                    .into_iter()
                    .map(|i| (i, self.dataset.query_to_one(self.query, i)))
                    .map(|(i, d)| (i, OrdNumber { number: d }));
            self.hits.extend(new_hits);
        });

        // let insider_cardinalities = self.insiders.iter().map(|g| g.c.cardinality()).sum::<usize>();
        // log::debug!(
        //     "Step {}: Insider cardinalities are {}, with another {} in hits ...",
        //     step,
        //     insider_cardinalities,
        //     self.hits.len()
        // );

        if self.straddlers.is_empty() || self.straddlers.iter().all(|g| g.c.is_leaf()) {
            self.insiders.drain(..).chain(self.straddlers.drain(..)).for_each(|g| {
                let new_hits =
                    g.c.indices()
                        // .into_par_iter()
                        .into_iter()
                        .map(|i| (i, self.dataset.query_to_one(self.query, i)))
                        .map(|(i, d)| (i, OrdNumber { number: d }));
                self.hits.extend(new_hits);
            });
            if self.hits.len() > self.k {
                let mut potential_ties = vec![self.hits.pop_max().unwrap()];
                while self.hits.len() >= self.k {
                    let item = self.hits.pop_max().unwrap();
                    if item.1.number < potential_ties.last().unwrap().1.number {
                        potential_ties.clear();
                    }
                    potential_ties.push(item);
                }
                self.hits.extend(potential_ties.drain(..));
            }
            self.is_refined = true;
            // log::debug!("Step {}: Sieve is refined! ...", step);
        } else {
            self.grains = self.insiders.drain(..).chain(self.straddlers.drain(..)).collect();
            let (leaves, non_leaves): (Vec<_>, Vec<_>) = self.grains.drain(..).partition(|g| g.c.is_leaf());

            // log::debug!(
            //     "Step {}: Of the straddlers, got {} leaves and {} non-leaves ...",
            //     step,
            //     leaves.len(),
            //     non_leaves.len()
            // );
            let children = non_leaves
                // .into_par_iter()
                .into_iter()
                .flat_map(|g| g.c.children())
                .map(|c| (c, self.dataset.query_to_one(self.query, c.arg_center())))
                .map(|(c, d)| Grain::new(c, d))
                .collect::<Vec<_>>();

            self.grains = leaves.into_iter().chain(children).collect();
            // log::debug!(
            //     "Step {}: Got {} grains for the next refinement step ...",
            //     step,
            //     self.grains.len()
            // );
        }
        self
    }

    pub fn refined_extract(&self) -> Vec<usize> {
        self.hits.iter().map(|(i, _)| *i).collect()
    }

    #[inline(never)]
    pub fn inexact_descend(mut self, _counter: usize) -> Self {
        // super::find_kth::find_kth(&mut self.grains, &mut self.cumulative_cardinalities, self.k);

        self.grains.sort();
        self.update_cumulative_cardinalities();

        let index = self.cumulative_cardinalities.iter().position(|&c| c >= self.k).unwrap();
        // log::info!(
        //     "Step: {}, Partition index is {}/{} and cardinality is {}/{} ...",
        //     _counter,
        //     index,
        //     self.grains.len(),
        //     self.cumulative_cardinalities[index],
        //     self.cumulative_cardinalities.last().unwrap()
        // );
        self.is_refined = self.cumulative_cardinalities[index] == self.k;
        if self.is_refined {
            self.grains.sort_by(|a, b| a.d_min.partial_cmp(&b.d_min).unwrap());
            self.grains = self.grains[..index].to_vec();
        } else {
            let threshold = self.grains[index].d_max;
            let _num_grains = self.grains.len();

            let (leaves, non_leaves): (Vec<_>, Vec<_>) = self
                .grains
                .into_iter()
                .filter(|g| !g.is_outside(threshold))
                .partition(|g| g.c.is_leaf());
            // log::info!(
            //     "Step: {}, leaves: {}, non-leaves: {}, excluded: {} ...",
            //     _counter,
            //     leaves.len(),
            //     non_leaves.len(),
            //     _num_grains - leaves.len() - non_leaves.len()
            // );

            // self.is_refined = non_leaves.is_empty() && leaves.iter().all(|g| g.d_max <= threshold);
            // self.is_refined = non_leaves.is_empty()
            //     || ((index == self.cumulative_cardinalities.len() - 1) && non_leaves.iter().all(|g| g.is_inside(threshold)));

            let (insiders, straddlers): (Vec<_>, Vec<_>) = non_leaves.into_iter().partition(|g| g.is_inside(threshold));
            // log::info!(
            //     "Step: {}, insiders: {}, straddlers: {} ...",
            //     _counter,
            //     insiders.len(),
            //     straddlers.len()
            // );

            let children = straddlers.into_iter().flat_map(|g| g.c.children()).collect::<Vec<_>>();
            let centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
            let distances = self.dataset.query_to_many(self.query, &centers);
            let children = children
                .into_iter()
                .zip(distances.into_iter())
                .map(|(c, d)| Grain::new(c, d));

            self.grains = leaves.into_iter().chain(insiders.into_iter()).chain(children).collect();
        }

        self
    }

    pub fn update_cumulative_cardinalities(&mut self) {
        // TODO: Take l, r indices and only update in between them
        self.cumulative_cardinalities = self
            .grains
            .iter()
            .scan(0, |acc, g| {
                *acc += g.c.cardinality();
                Some(*acc)
            })
            .collect();
    }

    pub fn update_guaranteed_cardinalities(&mut self) {
        self.guaranteed_cardinalities = self
            .grains
            .iter()
            .scan(0, |acc, g| {
                if g.is_inside(self.threshold) {
                    *acc += g.c.cardinality();
                } else {
                    *acc += 1;
                }
                Some(*acc)
            })
            .collect();
    }

    /// Returns `k` best hits from the sieve along with their distances from the
    /// query. If this method is called when the `is_refined` member is `true`,
    /// the result will have the best recall. If the `metric` in use obeys the
    /// triangle inequality, then the results will have perfect recall. If this
    /// method is called before the sieve has been fully refined, the results
    /// may have less than ideal recall.
    pub fn naive_extract(&self) -> Vec<usize> {
        self.grains.iter().flat_map(|g| g.c.indices()).collect()
    }

    // #[inline(never)]
    // fn update_guaranteed_cardinalities(&mut self) {
    //     // assumes that insiders and straddlers have been set.
    // }

    #[inline(never)]
    fn select_threshold(&mut self) -> usize {
        let kth_grain = if self.grains.len() < self.k {
            find_kth::find_kth_d_max(&mut self.grains, &mut self.cumulative_cardinalities, self.k)
        } else {
            find_kth::find_kth_d_max(&mut self.grains, &mut self.guaranteed_cardinalities, self.k)
        };

        self.update_guaranteed_cardinalities();
        self.update_cumulative_cardinalities();

        assert!(self.cumulative_cardinalities.last().copied().unwrap() >= self.k);
        self.threshold = kth_grain.0.d;
        kth_grain.1
    }

    #[inline(never)]
    pub fn shrink(mut self) -> Self {
        self.select_threshold();
        // log::info!("Initial threshold: {} ...", self.threshold);

        // let (mut insiders, mut straddlers) = (vec![], vec![]);
        // let mut counter = 0;

        loop {
            // log::info!("Shrink loop counter {} ...", counter);

            let num_grains = self.grains.len();
            let grains = self.grains;
            self.grains = vec![];
            [self.insiders, self.straddlers] = grain_filter(grains, self.threshold);

            // log::info!(
            //     "Shrink step {}: before break check: num_grains: {}, num_insiders: {}, num_straddlers: {} ...",
            //     counter,
            //     num_grains,
            //     self.insiders.len(),
            //     self.straddlers.len()
            // );

            if (self.insiders.len() + self.straddlers.len()) == num_grains {
                if self.straddlers.is_empty() || self.straddlers.iter().all(|g| g.c.is_leaf()) {
                    break;
                }

                let children = self
                    .straddlers
                    .into_iter()
                    .flat_map(|g| g.c.children())
                    .collect::<Vec<_>>();
                let centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
                let distances = self.dataset.query_to_many(self.query, &centers);
                let straddlers = children
                    .into_iter()
                    .zip(distances.into_iter())
                    .map(|(c, d)| Grain::new(c, d))
                    .collect();
                let [mut insiders, straddlers] = grain_filter(straddlers, self.threshold);

                self.insiders.append(&mut insiders);
                self.straddlers = straddlers;
            }

            // log::info!(
            //     "Shrink step {}: after break check: num_insiders: {}, num_straddlers: {} ...",
            //     counter,
            //     self.insiders.len(),
            //     self.straddlers.len()
            // );

            self.grains = self.insiders.drain(..).chain(self.straddlers.drain(..)).collect();
            if self.grains.len() < self.k {
                self.grains.sort_by(|a, b| a.ord_by_d_max(b));
            } else {
                self.grains.sort_by(|a, b| a.ord_by_d(b));
            }

            self.update_guaranteed_cardinalities();

            // let last = self.guaranteed_cardinalities.last().copied().unwrap_or(0);
            // self.guaranteed_cardinalities
            //     .extend(last..(last + self.straddlers.len()));

            // let index = self.guaranteed_cardinalities.iter().position(|c| *c >= self.k).unwrap();

            // let old_threshold = self.threshold;
            self.select_threshold();
            // let index = self.select_threshold();
            // log::info!("Shrink step {}: threshold index {} ...", counter, index);
            // assert!(old_threshold < self.threshold);

            // let threshold = if index < self.insiders.len() {
            //     let d_maxs = self.insiders.iter().map(|g| g.d_max).collect::<Vec<_>>();
            //     helpers::arg_max(&d_maxs).1
            // } else {
            //     self.straddlers[index - self.insiders.len()].d
            // };
            // self.threshold = threshold;
            // log::info!("Step {}: new threshold {} ...", counter, self.threshold);

            // self.grains = self.insiders.drain(..).chain(self.straddlers.drain(..)).collect();

            // log::info!(
            //     "Shrink step {}: after resetting grains: num_grains: {} ...",
            //     counter,
            //     self.grains.len()
            // );
            // log::info!("");

            // counter += 1;
        }

        self.is_refined = true;

        self
    }

    #[inline(never)]
    pub fn extract(&mut self) -> Vec<usize> {
        let hits = self.insiders.iter().flat_map(|g| g.c.indices()).collect::<Vec<_>>();
        let distances = self.dataset.query_to_many(self.query, &hits);

        let mut pq = priority_queue::DoublePriorityQueue::new();
        pq.extend(
            hits.into_iter()
                .zip(distances.into_iter())
                .map(|(i, d)| (i, OrdNumber { number: d })),
        );

        while !self.straddlers.is_empty() {
            let candidate = self.straddlers.pop().unwrap().c;
            let indices = candidate.indices();
            let distances = self.dataset.query_to_many(self.query, &indices);
            pq.extend(
                indices
                    .into_iter()
                    .zip(distances.into_iter())
                    .map(|(i, d)| (i, OrdNumber { number: d })),
            );

            let mut potential_ties = vec![];
            while pq.len() > self.k {
                // pq.pop_max();
                potential_ties.push(pq.pop_max().unwrap());
            }

            let threshold = pq.peek_max().unwrap().1.number;
            pq.extend(potential_ties.into_iter().filter(|(_, d)| d.number <= threshold));

            self.straddlers = self.straddlers.drain(..).filter(|g| !g.is_outside(threshold)).collect();
        }

        let mut potential_ties = vec![];
        while pq.len() > self.k {
            // pq.pop_max();
            potential_ties.push(pq.pop_max().unwrap());
        }

        let threshold = pq.peek_max().unwrap().1.number;
        pq.extend(potential_ties.into_iter().filter(|(_, d)| d.number <= threshold));

        pq.into_iter().map(|(i, _)| i).collect()
    }
}
