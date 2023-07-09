use distances::Number;

use crate::{
    cluster::{Cluster, Tree},
    dataset::Dataset,
};

#[allow(dead_code)]
pub struct KnnSieve<'a, T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> {
    tree: &'a Tree<T, U, D>,
    query: T,
    k: usize,
    layer: Vec<&'a Cluster<T, U>>,
    leaves: Vec<Grain<'a, T, U>>,
    is_refined: bool,
    hits: priority_queue::DoublePriorityQueue<usize, OrdNumber<U>>,
}

impl<'a, T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> KnnSieve<'a, T, U, D> {
    pub fn new(tree: &'a Tree<T, U, D>, query: T, k: usize) -> Self {
        Self {
            layer: vec![tree.root()],
            tree,
            query,
            k,
            leaves: Vec::new(),
            is_refined: false,
            hits: Default::default(),
        }
    }

    pub fn is_refined(&self) -> bool {
        self.is_refined
    }

    pub fn refine_step(&mut self) {
        let data = self.tree.data();
        let distances = self
            .layer
            .iter()
            .map(|c| c.distance_to_instance(data, self.query))
            .collect::<Vec<_>>();

        let mut grains = self
            .layer
            .drain(..)
            .zip(distances.iter())
            .flat_map(|(c, &d)| {
                if c.is_singleton() {
                    vec![Grain::new(c, d, c.cardinality)]
                } else {
                    let g = Grain::new(c, d, 1);
                    let g_max = Grain::new(c, d + c.radius, c.cardinality - 1);
                    vec![g, g_max]
                }
            })
            .chain(self.leaves.drain(..))
            .collect::<Vec<_>>();

        let i = Grain::partition_kth(&mut grains, self.k);
        // let threshold = grains[i].d;
        let num_guaranteed = grains[..=i].iter().map(|g| g.multiplicity).sum::<usize>();

        // TODO: Filter grains by being outside the threshold
        // partition into insiders and straddlers
        // descend into straddlers

        assert!(
            num_guaranteed >= self.k,
            "Too few guarantees: {num_guaranteed} vs {}.",
            self.k
        );

        todo!()
    }

    pub fn extract(&self) -> Vec<(usize, U)> {
        todo!()
    }
}

#[allow(dead_code)]
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
struct OrdNumber<U: Number>(U);

impl<U: Number> PartialEq for OrdNumber<U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<U: Number> Eq for OrdNumber<U> {}

impl<U: Number> PartialOrd for OrdNumber<U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<U: Number> Ord for OrdNumber<U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
