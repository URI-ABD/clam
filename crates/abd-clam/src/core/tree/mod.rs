//! A `Tree` of `Cluster`s.

use distances::Number;
use rayon::prelude::*;

use super::{
    cluster::{ParPartition, Partition},
    dataset::ParDataset,
    metric::ParMetric,
    Cluster, Dataset, Metric,
};

/// A `Tree` of `Cluster`s.
pub struct Tree<T: Number, C: Cluster<T>> {
    /// The `Cluster`s of the `Tree` are stored in `levels` where the first
    /// `Vec` contains the `Cluster`s at the first level, the second `Vec`
    /// contains the `Cluster`s at the second level, and so on.
    ///
    /// Each `Cluster` is represented by a tuple `(node, a, b, c, d)` where
    /// `node` is the `Cluster` and `a` and `b` define the range of indices of
    /// the children of the `Cluster` in the next level.
    levels: Vec<Vec<(C, usize, usize)>>,
    /// The diameter of the root `Cluster` in the `Tree`.
    diameter: T,
}

impl<T: Number, C: Cluster<T>> From<C> for Tree<T, C> {
    fn from(mut c: C) -> Self {
        let diameter = c.radius().double();
        if c.is_leaf() {
            Self {
                levels: vec![vec![(c, 0, 0)]],
                diameter,
            }
        } else {
            let mut child_trees = c
                .take_children()
                .into_iter()
                .map(|child| Self::from(*child))
                .collect::<Vec<_>>();
            let n_children = child_trees.len();

            let rest = child_trees.split_off(1);
            let first = child_trees
                .pop()
                .unwrap_or_else(|| unreachable!("We checked that the `Cluster` is not a leaf."));
            let mut tree = rest.into_iter().fold(first, Self::merge);

            tree.levels.insert(0, vec![(c, 0, n_children)]);
            tree.diameter = diameter;

            tree
        }
    }
}

impl<T: Number, C: Cluster<T>> Tree<T, C> {
    /// Returns the root `Cluster` of the `Tree`.
    pub fn root(&self) -> (&C, usize, usize) {
        self.levels
            .first()
            .and_then(|level| level.first().map(|(node, a, b)| (node, *a, *b)))
            .unwrap_or_else(|| unreachable!("The `Tree` is empty."))
    }

    /// Returns the `Cluster` at the given `depth` and `index`.
    pub fn get(&self, depth: usize, index: usize) -> Option<&C> {
        self.levels
            .get(depth)
            .and_then(|level| level.get(index).map(|(node, _, _)| node))
    }

    /// Finds the `Cluster` in the `Tree` that is equal to the given `Cluster`.
    ///
    /// If the `Cluster` is found, the method returns:
    ///
    /// - The depth of the `Cluster` in the `Tree`.
    /// - The index of the `Cluster` in its level the `Tree`.
    /// - The start index of the children of the `Cluster` in the next level.
    /// - The end index of the children of the `Cluster` in the next level.
    pub fn find(&self, c: &C) -> Option<(usize, usize, usize, usize)> {
        self.levels
            .get(c.depth())
            .and_then(|level| {
                let pos = level.iter().position(|(node, _, _)| node == c);
                pos.map(|index| (index, &level[index]))
            })
            .map(|(index, (_, a, b))| (c.depth(), index, *a, *b))
    }

    /// Returns the the children of the `Cluster` at the given `depth` and
    /// `index`.
    pub fn children_of(&self, depth: usize, index: usize) -> Vec<(&C, usize, usize)> {
        self.levels
            .get(depth)
            .and_then(|level| {
                level.get(index).and_then(|&(_, a, b)| {
                    self.levels
                        .get(depth + 1)
                        .map(|level| level[a..b].iter().map(|(node, a, b)| (node, *a, *b)).collect())
                })
            })
            .unwrap_or_default()
    }

    /// Returns the diameter of the `Tree`.
    pub const fn diameter(&self) -> T {
        self.diameter
    }

    /// Returns the levels of the `Tree`.
    pub fn levels(&self) -> &[Vec<(C, usize, usize)>] {
        &self.levels
    }

    /// Returns the clusters in the tree in breadth-first order.
    pub fn bft(&self) -> impl Iterator<Item = &C> {
        self.levels
            .iter()
            .flat_map(|level| level.iter().map(|(node, _, _)| node))
    }

    /// Merges the `Tree` with an other `Tree`, consuming both `Tree`s and
    /// keeping the diameter of first `Tree`.
    fn merge(mut self, mut other: Self) -> Self {
        let mut s_levels = core::mem::take(&mut self.levels);
        let mut o_levels = core::mem::take(&mut other.levels);

        let zipped_levels = s_levels.iter_mut().skip(1).zip(o_levels.iter_mut()).collect::<Vec<_>>();
        for (s_level, o_level) in zipped_levels.into_iter().rev() {
            let n = s_level.len();
            for (_, a, b) in o_level {
                *a += n;
                *b += n;
            }
        }

        let remaining_levels = if s_levels.len() < o_levels.len() {
            o_levels.split_off(s_levels.len())
        } else {
            s_levels.split_off(o_levels.len())
        };
        self.levels = s_levels
            .into_iter()
            .zip(o_levels)
            .map(|(mut s_level, mut o_level)| {
                s_level.append(&mut o_level);
                s_level
            })
            .chain(remaining_levels)
            .collect();

        self
    }
}

impl<T: Number, C: Partition<T>> Tree<T, C> {
    /// Constructs a `Tree` from the given `data` and `metric`.
    ///
    /// # Errors
    ///
    /// Any error from `C::new` is propagated.
    pub fn new<I, D: Dataset<I>, M: Metric<I, T>, F: Fn(&C) -> bool>(
        data: &D,
        metric: &M,
        criteria: &F,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let root = C::new(data, metric, &indices, 0, seed)?;
        let diameter = root.radius().double();

        let mut levels = vec![vec![(root, 0, 0)]];
        loop {
            let mut last_level = levels
                .last_mut()
                .unwrap_or_else(|| unreachable!("We inserted the root."))
                .iter_mut()
                .filter_map(|(node, a, b)| if criteria(node) { Some((node, a, b)) } else { None })
                .collect::<Vec<_>>();

            let children = last_level
                .iter_mut()
                .map(|(node, _, _)| {
                    node.partition_once(data, metric, seed)
                        .into_iter()
                        .map(|child| *child)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            if children.is_empty() {
                break;
            }

            let mut next_level = Vec::new();
            for ((_, a, b), children) in last_level.into_iter().zip(children) {
                *a = next_level.len();
                *b = next_level.len() + children.len();
                next_level.extend(children.into_iter().map(|child| (child, 0, 0)));
            }
            levels.push(next_level);
        }

        Ok(Self { levels, diameter })
    }
}

impl<T: Number, C: ParPartition<T>> Tree<T, C> {
    /// Parallel version of [`Tree::new`](crate::core::tree::Tree::new).
    ///
    /// # Errors
    ///
    /// Any error from `C::par_new` is propagated.
    pub fn par_new<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>, F: (Fn(&C) -> bool) + Send + Sync>(
        data: &D,
        metric: &M,
        criteria: &F,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let root = C::par_new(data, metric, &indices, 0, seed)?;
        let diameter = root.radius().double();

        let mut levels = vec![vec![(root, 0, 0)]];
        loop {
            let mut last_level = levels
                .last_mut()
                .unwrap_or_else(|| unreachable!("We inserted the root."))
                .into_par_iter()
                .filter_map(|(node, a, b)| if criteria(node) { Some((node, a, b)) } else { None })
                .collect::<Vec<_>>();

            let children = last_level
                .par_iter_mut()
                .map(|(node, _, _)| {
                    node.par_partition_once(data, metric, seed)
                        .into_iter()
                        .map(|child| *child)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            if children.is_empty() {
                break;
            }

            let mut next_level = Vec::new();
            for ((_, a, b), children) in last_level.into_iter().zip(children) {
                *a = next_level.len();
                *b = next_level.len() + children.len();
                next_level.extend(children.into_iter().map(|child| (child, 0, 0)));
            }
            levels.push(next_level);
        }

        Ok(Self { levels, diameter })
    }
}
