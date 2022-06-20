use crate::prelude::*;

use priority_queue::PriorityQueue;

#[derive(Clone, Copy)]
pub struct OrderedNumber<U: Number>(U);

impl<U: Number> PartialEq for OrderedNumber<U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<U: Number> Eq for OrderedNumber<U> {}

impl<U: Number> PartialOrd for OrderedNumber<U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<U: Number> Ord for OrderedNumber<U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Greater)
    }
}

type ClusterQueue<'a, T, U> = PriorityQueue<&'a Cluster<'a, T, U>, OrderedNumber<U>>;
type InstanceQueue<'a, U> = PriorityQueue<usize, OrderedNumber<U>>;

pub struct KnnQueue<'a, T: Number, U: Number> {
    by_delta_0: ClusterQueue<'a, T, U>,
    by_delta_1: ClusterQueue<'a, T, U>,
    by_delta_2: ClusterQueue<'a, T, U>,
    hits: InstanceQueue<'a, U>,
}

impl<'a, T: Number, U: Number> KnnQueue<'a, T, U> {
    pub fn new(clusters_distances: &'a [(&Cluster<T, U>, U)]) -> Self {
        let by_delta_0 = PriorityQueue::from_iter(clusters_distances.iter().map(|(c, d)| (*c, OrderedNumber(*d))));

        let by_delta_1 = PriorityQueue::from_iter(
            clusters_distances
                .iter()
                .map(|(c, d)| (*c, OrderedNumber(c.radius() + *d))),
        );

        let by_delta_2 = PriorityQueue::from_iter(clusters_distances.iter().map(|(c, d)| {
            (
                *c,
                OrderedNumber(if c.radius() > *d { c.radius() - *d } else { U::zero() }),
            )
        }));

        Self {
            by_delta_0,
            by_delta_1,
            by_delta_2,
            hits: PriorityQueue::new(),
        }
    }

    pub fn by_delta_0(&self) -> Vec<&Cluster<T, U>> {
        self.by_delta_0.clone().into_sorted_iter().map(|(c, _)| c).collect()
    }

    pub fn by_delta_1(&self) -> Vec<&Cluster<T, U>> {
        self.by_delta_1.clone().into_sorted_iter().map(|(c, _)| c).collect()
    }

    pub fn by_delta_2(&self) -> Vec<&Cluster<T, U>> {
        self.by_delta_2.clone().into_sorted_iter().map(|(c, _)| c).collect()
    }

    pub fn hits(&self) -> Vec<usize> {
        self.hits.clone().into_sorted_vec()
    }
}
