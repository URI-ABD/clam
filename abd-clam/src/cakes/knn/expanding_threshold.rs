//! Search function and helper functions for knn with expanding threshold.


use core::cmp::Ordering;

use distances::Number;

use crate::{Cluster, Dataset, Tree};

pub fn search<T, U, D>(tree: &Tree<T, U, D>, query: T, k: usize) -> Vec<(usize, U)>
where
    T: Send + Sync + Copy,
    U: Number,
    D: Dataset<T, U>,
{
    let mut candidates = priority_queue::PriorityQueue::<&Cluster<T, U>, RevNumber<U>>::new();
    let mut hits = priority_queue::PriorityQueue::<usize, OrdNumber<U>>::new();
    let d = tree.root().distance_to_instance(tree.data(), query);
    candidates.push(tree.root(), RevNumber(d_min(tree.root(), d)));

    // stop if we have enough hits and the farthest hit is closer than the closest cluster by delta_min.
    while !(hits.len() >= k
        && (candidates.is_empty() || hits.peek().unwrap().1.0 < candidates.peek().unwrap().1 .0))
    {
        pop_till_leaf(tree, query, &mut candidates);
        leaf_into_hits(tree, query, &mut hits, &mut candidates);
        trim_hits(k, &mut hits);
    }
    assert!(hits.len() >= k);

    hits.into_iter().map(|(i, OrdNumber(d))| (i, d)).collect()
}


fn d_min<T: Send + Sync + Copy, U: Number>(c: &Cluster<T, U>, d: U) -> U {
    if d < c.radius {
        U::zero()
    } else {
        d - c.radius
    }
}

// pop from the top of `candidates` until the top candidate is a leaf cluster.
fn pop_till_leaf<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>>(tree: &Tree<T, U, D>, query: T, candidates: &mut priority_queue::PriorityQueue<&Cluster<T, U>, RevNumber<U>>) {
    while !candidates.peek().unwrap().0.is_leaf() {
        let [l, r] = candidates.pop().unwrap().0.children().unwrap();
        let [dl, dr] = [
            l.distance_to_instance(tree.data(), query),
            r.distance_to_instance(tree.data(), query),
        ];
        candidates.push(l, RevNumber(d_min(l, dl)));
        candidates.push(r, RevNumber(d_min(r, dr)));
    }
}

// pop a single leaf from the top of candidates and add those points to hits.
fn leaf_into_hits<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>>(
    tree: &Tree<T, U, D>,
    query: T,
    hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>,
    candidates: &mut priority_queue::PriorityQueue<&Cluster<T, U>, RevNumber<U>>,
) {
    let (leaf, RevNumber(d)) = candidates.pop().unwrap();
    let is = leaf.indices(tree.data());
    let ds = if leaf.is_singleton() {
        vec![d; is.len()]
    } else {
        tree.data().query_to_many(query, is)
    };
    is.iter().zip(ds.into_iter()).for_each(|(&i, d)| {
        hits.push(i, OrdNumber(d));
    });
}

// reduce hits down to k elements, including ties for the kth farthest element.
fn trim_hits<U: Number>(k: usize, hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>) {
    if hits.len() > k {
        let mut potential_ties = vec![hits.pop().unwrap()];
        while hits.len() >= k {
            let item = hits.pop().unwrap();
            if item.1< potential_ties.last().unwrap().1 {
                potential_ties.clear();
            }
            potential_ties.push(item);
        }
        hits.extend(potential_ties.drain(..));
    }
}



/// Field by which we rank elements in priority queue of hits.
#[derive(Debug)]
pub struct OrdNumber<U: Number> {
    /// The number we use to rank elements (distance to query).
    number: U,
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
        self.partial_cmp(other).unwrap_or_else(|| {
            unreachable!(
                "All hits are instances, and
        therefore each hit has a distance from the query. Since all hits' distances to the
        query will be represented by the same type, we can always compare them."
            )
        })
    }
}