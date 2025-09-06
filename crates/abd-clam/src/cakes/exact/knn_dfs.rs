//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use crate::{
    Cluster, DistanceValue, Tree,
    cakes::{ParSearch, Search, d_max, d_min},
    utils::SizedHeap,
};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnDfs(pub usize);

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnDfs {
    fn name(&self) -> String {
        format!("KnnDfs(k={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();

        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree.distances_to_items_in_cluster(query, root).collect();
        }

        let mut candidates = SizedHeap::<&Cluster<T, A>, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<usize, T>::new(Some(self.0));

        let d = tree.distance_to_center(query, root);
        hits.push((root.center_index(), d));
        candidates.push((root, Reverse((d_min(root, d), d_max(root, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, _) = pop_till_leaf(query, tree, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            leaf_into_hits(query, tree, &mut hits, leaf, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);
            if hits.is_full() && max_h < min_c {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        hits.take_items().collect()
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for KnnDfs
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        // For now, just call the single-threaded search.
        self.search(tree, query)
    }
}

/// Pop candidates until the top candidate is a leaf, then pop and return that leaf along with its minimum distance from the query.
///
/// The user must ensure that `candidates` is non-empty before calling this function.
#[allow(clippy::type_complexity)]
pub fn pop_till_leaf<'a, Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T>(
    query: &I,
    tree: &'a Tree<Id, I, T, A, M>,
    candidates: &mut SizedHeap<&'a Cluster<T, A>, Reverse<(T, T, T)>>,
    hits: &mut SizedHeap<usize, T>,
) -> (&'a Cluster<T, A>, T, usize) {
    profi::prof!("KnnDfs::pop_till_leaf");

    let mut distance_computations = 0;
    while candidates
        .peek()
        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(cluster, _)| !cluster.is_leaf())
    {
        profi::prof!("pop-while-not-leaf");

        candidates.pop().and_then(|(parent, _)| tree.children_of(parent)).map_or_else(
            || unreachable!("Top candidate is a parent."),
            |children| {
                distance_computations += children.len();

                for child in children {
                    let ci = child.center_index();
                    let d = tree.distance_to(query, ci);
                    hits.push((ci, d));
                    candidates.push((child, Reverse((d_min(child, d), d_max(child, d), d))));
                }
            },
        );
    }

    let (leaf, d) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(leaf, Reverse((_, _, d)))| (leaf, d));
    (leaf, d, distance_computations)
}

/// Given a leaf cluster, compute the distance from the query to each item in the leaf and push them onto `hits`.
///
/// Returns the number of distance computations performed, excluding the distance to the center (which is already known).
pub fn leaf_into_hits<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T>(
    query: &I,
    tree: &Tree<Id, I, T, A, M>,
    hits: &mut SizedHeap<usize, T>,
    leaf: &Cluster<T, A>,
    d: T,
) -> usize {
    profi::prof!("KnnDfs::leaf_into_hits");

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are exactly `d` from the query.
        hits.extend(leaf.items_indices().skip(1).map(|i| (i, d)));
        0
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute the distance from the query to each item in the leaf.
        hits.extend(tree.distances_to_items_in_subtree(query, leaf));
        leaf.cardinality() - 1 // We already knew the distance to the center.
    }
}
