use std::cmp::Ordering;

use distances::Number;

use crate::cluster::{Cluster, Tree};
use crate::dataset::Dataset;

#[allow(dead_code)]
pub struct KnnSieve<'a, T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> {
    tree: &'a Tree<T, U, D>,
    query: T,
    k: usize,
    grains: Vec<Grain<'a, T, U>>,
    is_refined: bool,
    hits: priority_queue::DoublePriorityQueue<usize, OrdNumber<U>>,
    layer: Vec<&'a Cluster<T, U>>,
}

impl<'a, T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> KnnSieve<'a, T, U, D> {
    pub fn new(tree: &'a Tree<T, U, D>, query: T, k: usize) -> Self {
        Self {
            tree,
            query,
            k,
            grains: Vec::new(),
            is_refined: false,
            hits: Default::default(),
            layer: vec![tree.root()],
        }
    }

    pub fn initialize_grains(&mut self) {
        let mut layer = vec![self.tree.root()];

        let distances = layer
            .iter()
            .map(|c| c.distance_to_instance(self.tree.data(), self.query))
            .collect::<Vec<_>>();

        self.grains = layer
            .drain(..)
            .zip(distances.iter())
            .flat_map(|(c, &d)| [Grain::new(c, d, 1), Grain::new(c, d + c.radius, c.cardinality - 1)])
            .collect::<Vec<_>>();
    }

    pub fn is_refined(&self) -> bool {
        self.is_refined
    }

    pub fn refine_step(&mut self) {
        let i = Grain::partition_kth(&mut self.grains, self.k);
        let threshold = self.grains[i].d;

        // Filters grains by being outside the threshold.
        // Ties are added to hits together; we will never remove too many instances here
        // because our choice of threshold guarantees enough instances.
        while !self.hits.is_empty() && self.hits.peek_max().unwrap().1.number > threshold {
            self.hits.pop_max().unwrap();
        }

        // partition into insiders and straddlers
        // where we filter for grains being outside the threshold could be made more
        // efficient by leveraging the fact that partition already puts items on the correct
        // side of the threshold elememultiplicitynt
        let (mut insiders, mut straddlers): (Vec<_>, Vec<_>) = self
            .grains
            .drain(..)
            .filter(|g| !Grain::is_outside(g, threshold))
            .partition(|g| Grain::is_inside(g, threshold));

        // distinguish between those        println!("insiders are: {:?}", returnable_insiders);
        // insiders we won't partition further and those we will
        // add instances from insiders we won't further partition to hits
        let (small_insiders, big_insiders): (Vec<_>, Vec<_>) = insiders
            .drain(..)
            .partition(|g| (g.c.cardinality <= self.k) || g.c.is_leaf()); // TODO: fix this so that only Grains with cluster CARDINALITY, not multiplicity <= k stay
        insiders = big_insiders;
        // need to do a similar thing for the centers that ONLY adds center point, which is why
        // I want arg center again.
        // Right now instead of adding just the center to hits it adds everything in the cluster
        small_insiders.into_iter().for_each(|g| {
            let new_hits = self.tree.indices_of(g.c).iter().map(|&i| {
                (
                    i,
                    OrdNumber {
                        number: self.tree.data().query_to_one(self.query, i),
                    },
                )
            });
            self.hits.extend(new_hits);
        });

        // descend into straddlers

        // If there are no straddlers or all of the straddlers are leaves, then the grains in insiders and straddlers
        // are added to hits. If there are more than k hits, we repeatedly remove the furthest instance in hits until
        // there are either k hits left or more than k hits with some ties (d            .iter()
        // the same distance from the query)
        //tree.rs
        // If straddlers is not empty nor all leaves, partition non-leaves into children
        if straddlers.is_empty() || straddlers.iter().all(|g| g.c.is_leaf()) {
            insiders.drain(..).chain(straddlers.drain(..)).for_each(|g| {
                let new_hits = self.tree.indices_of(g.c).iter().map(|&i| {
                    (
                        i,
                        OrdNumber {
                            number: self.tree.data().query_to_one(self.query, i),
                        },
                    )
                });

                self.hits.extend(new_hits);
            });

            if self.hits.len() > self.k {
                let mut potential_ties = vec![self.hits.pop_max().unwrap()];
                while self.hits.len() >= self.k {
                    let item = self.hits.pop_max().unwrap();
                    if item.1.number <= potential_ties.last().unwrap().1.number {
                        potential_ties.clear();
                    }
                    potential_ties.push(item);
                }
                self.hits.extend(potential_ties.drain(..));
            }

            self.is_refined = true;
        } else {
            self.grains = insiders.drain(..).chain(straddlers.drain(..)).collect();
            let (leaves, non_leaves): (Vec<_>, Vec<_>) = self.grains.drain(..).partition(|g| g.c.is_leaf());
            let (partitionable_grains, np_grains): (Vec<_>, Vec<_>) =
                non_leaves.into_iter().partition(|g| g.multiplicity > 1);
            let children = partitionable_grains
                .iter()
                .flat_map(|g| g.c.children().unwrap())
                .map(|c| (c, c.distance_to_instance(self.tree.data(), self.query)))
                .flat_map(|(c, d)| [Grain::new(c, d, 1), Grain::new(c, d + c.radius, c.cardinality - 1)]);

            self.grains = leaves.into_iter().chain(children).chain(np_grains).collect();
        }
    }

    pub fn extract(&self) -> Vec<(usize, U)> {
        self.hits.iter().map(|(&i, d)| (i, d.number)).collect()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Grain<'a, T: Send + Sync + Copy, U: Number> {
    t: std::marker::PhantomData<T>,
    c: &'a Cluster<T, U>,
    d: U,
    multiplicity: usize,
}

impl<'a, T: Send + Sync + Copy, U: Number> Grain<'a, T, U> {
    fn new(c: &'a Cluster<T, U>, d: U, multiplicity: usize) -> Self {
        let t = Default::default();
        Self { t, c, d, multiplicity }
    }

    #[allow(dead_code)]
    /// A Grain is "straddling" the threshold if it could have at least one point which is a distance less than the
    /// threshold distance from the query and at least one point which is a distance greater than or equal to the
    /// threshold distance from the query, i.e., if it is neither inside nor outside the threshold.
    fn is_straddling(&self, threshold: U) -> bool {
        !(self.is_inside(threshold) || self.is_outside(threshold))
    }

    /// A Grain is "inside" the threshold if the furthest, worst-case possible point is at most as far as
    /// threshold distance from the query, i.e., if d_max is greater than or equal to the threshold distance
    fn is_inside(&self, threshold: U) -> bool {
        if self.multiplicity == 1 {
            return self.d <= threshold;
        }
        self.d < threshold
    }

    /// A Grain is "outside" the threshold if the closest, best-case possible point is further than
    /// the threshold distance to the query, i.e., if d_min is less than the threshold distance
    fn is_outside(&self, threshold: U) -> bool {
        if self.multiplicity == 1 {
            self.d > threshold
        } else {
            let d_min = if self.d < self.c.radius + self.c.radius {
                U::zero()
            } else {
                self.d - self.c.radius - self.c.radius
            };

            d_min > threshold
        }
    }
    fn partition_kth(grains: &mut [Self], k: usize) -> usize {
        let i = Self::_partition_kth(grains, k, 0, grains.len() - 1);
        let t = grains[i].d;

        let mut b = i;
        for a in (i + 1)..(grains.len()) {
            if grains[a].d == t {
                b += 1;
                grains.swap(a, b);
            }
        }

        b
    }

    fn _partition_kth(grains: &mut [Self], k: usize, l: usize, r: usize) -> usize {
        if l >= r {
            std::cmp::min(l, r)
        } else {
            let p = Self::_partition(grains, l, r);
            let guaranteed = grains
                .iter()
                .scan(0, |acc, g| {
                    *acc += g.multiplicity;
                    Some(*acc)
                })
                .collect::<Vec<_>>();

            let num_g = guaranteed[p];

            match num_g.cmp(&k) {
                std::cmp::Ordering::Less => Self::_partition_kth(grains, k, p + 1, r),
                std::cmp::Ordering::Equal => p,
                std::cmp::Ordering::Greater => {
                    if (p > 0) && (guaranteed[p - 1] > k) {
                        Self::_partition_kth(grains, k, l, p - 1)
                    } else {
                        p
                    }
                }
            }
        }
    }

    fn _partition(grains: &mut [Self], l: usize, r: usize) -> usize {
        let pivot = (l + r) / 2;
        grains.swap(pivot, r);

        let (mut a, mut b) = (l, l);
        while b < r {
            if grains[b].d <= grains[r].d {
                grains.swap(a, b);
                a += 1;
            }
            b += 1;
        }

        grains.swap(a, r);

        a
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
