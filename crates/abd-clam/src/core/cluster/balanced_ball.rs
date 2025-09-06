//! A `BalancedBall` is a data structure that represents a balanced binary tree.

use num::traits::{FromBytes, ToBytes};
use rayon::prelude::*;

use crate::{dataset::ParDataset, Dataset};

use super::{Ball, Cluster, DistanceValue, ParCluster, ParPartition, Partition};

/// A `BalancedBall` is a data structure that represents a balanced binary tree.
#[derive(Clone)]
#[must_use]
pub struct BalancedBall<T: DistanceValue>(Ball<T>, Vec<Box<Self>>);

impl<T: DistanceValue> BalancedBall<T> {
    /// Converts the `BalancedBall` into a `Ball`.
    pub fn into_ball(mut self) -> Ball<T> {
        if !self.1.is_empty() {
            let children = self.1.into_iter().map(|c| c.into_ball()).map(Box::new).collect();
            self.0.set_children(children);
        }
        self.0
    }
}

impl<T: DistanceValue + core::fmt::Debug> core::fmt::Debug for BalancedBall<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BalancedBall")
            .field("depth", &self.depth())
            .field("cardinality", &self.cardinality())
            .field("radius", &self.radius())
            .field("lfd", &self.lfd())
            .field("arg_center", &self.arg_center())
            .field("arg_radial", &self.arg_radial())
            .field("indices", &self.indices())
            .field("children", &!self.is_leaf())
            .finish()
    }
}

impl<T: DistanceValue> PartialEq for BalancedBall<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<T: DistanceValue> Eq for BalancedBall<T> {}

impl<T: DistanceValue> PartialOrd for BalancedBall<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: DistanceValue> Ord for BalancedBall<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T: DistanceValue> Cluster<T> for BalancedBall<T> {
    fn depth(&self) -> usize {
        self.0.depth()
    }

    fn cardinality(&self) -> usize {
        self.0.cardinality()
    }

    fn arg_center(&self) -> usize {
        self.0.arg_center()
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.0.set_arg_center(arg_center);
    }

    fn radius(&self) -> T {
        self.0.radius()
    }

    fn arg_radial(&self) -> usize {
        self.0.arg_radial()
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.0.set_arg_radial(arg_radial);
    }

    fn lfd(&self) -> f32 {
        self.0.lfd()
    }

    fn contains(&self, idx: usize) -> bool {
        self.0.contains(idx)
    }

    fn indices(&self) -> Vec<usize> {
        self.0.indices()
    }

    fn set_indices(&mut self, indices: &[usize]) {
        self.0.set_indices(indices);
    }

    fn take_indices(&mut self) -> Vec<usize> {
        self.0.take_indices()
    }

    fn children(&self) -> Vec<&Self> {
        self.1.iter().map(AsRef::as_ref).collect()
    }

    fn children_mut(&mut self) -> Vec<&mut Self> {
        self.1.iter_mut().map(AsMut::as_mut).collect()
    }

    fn set_children(&mut self, children: Vec<Box<Self>>) {
        self.1 = children;
    }

    fn take_children(&mut self) -> Vec<Box<Self>> {
        core::mem::take(&mut self.1)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.0.is_descendant_of(&other.0)
    }
}

impl<T: DistanceValue + Send + Sync> ParCluster<T> for BalancedBall<T> {
    fn par_indices(&self) -> impl rayon::prelude::ParallelIterator<Item = usize> {
        self.0.par_indices()
    }
}

impl<T: DistanceValue> Partition<T> for BalancedBall<T> {
    fn new<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        data: &D,
        metric: &M,
        indices: &[usize],
        depth: usize,
    ) -> Result<Self, String> {
        Ball::new(data, metric, indices, depth).map(|ball| Self(ball, Vec::new()))
    }

    fn find_extrema<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(&self, data: &D, metric: &M) -> Vec<usize> {
        self.0.find_extrema(data, metric)
    }

    #[allow(clippy::similar_names)]
    fn split_by_extrema<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &self,
        data: &D,
        metric: &M,
        extrema: &[usize],
    ) -> Vec<Vec<usize>> {
        let [l, r] = [extrema[0], extrema[1]];
        let lr = data.one_to_one(l, r, metric);

        let items = self
            .indices()
            .into_iter()
            .filter(|&i| !(i == l || i == r))
            .collect::<Vec<_>>();

        // Find the distances from each extremum to each item.
        let l_distances = data.one_to_many(l, &items, metric);
        let r_distances = data.one_to_many(r, &items, metric);

        let child_stacks = {
            let lr = lr
                .to_f32()
                .unwrap_or_else(|| unreachable!("Cannot convert distance to f32"));

            // Find the distance from `l` to each item projected onto the line
            // connecting `l` and `r`.
            let lr_distances = {
                let mut lr_distances = l_distances
                    .map(|(a, d)| {
                        (
                            a,
                            d.to_f32()
                                .unwrap_or_else(|| unreachable!("Cannot convert distance to f32")),
                        )
                    })
                    .zip(r_distances.map(|(_, d)| {
                        d.to_f32()
                            .unwrap_or_else(|| unreachable!("Cannot convert distance to f32"))
                    }))
                    .map(|((a, al), ar)| {
                        let cos = ar.mul_add(-ar, lr.mul_add(lr, al.powi(2))) / (2.0 * al * lr);
                        (a, al * cos)
                    })
                    .collect::<Vec<_>>();
                lr_distances.sort_by(|(_, a), (_, b)| a.total_cmp(b));
                lr_distances
                    .into_iter()
                    .map(|(a, d)| {
                        (
                            a,
                            T::from_f32(d).unwrap_or_else(|| {
                                unreachable!("Cannot convert f32 to {}", core::any::type_name::<T>())
                            }),
                        )
                    })
                    .collect::<Vec<_>>()
            };

            // Half of the items will be assigned to the left child and the
            // other half to the right child.
            let mid = if lr_distances.len() % 2 == 0 {
                lr_distances.len() / 2
            } else {
                1 + lr_distances.len() / 2
            };
            let (ls, rs) = lr_distances.split_at(mid);
            let l_stack = core::iter::once((l, T::zero()))
                .chain(ls.iter().copied())
                .collect::<Vec<_>>();
            let r_stack = core::iter::once((r, T::zero()))
                .chain(rs.iter().copied())
                .collect::<Vec<_>>();

            vec![l_stack, r_stack]
        };

        child_stacks
            .into_iter()
            .map(|stack| {
                let (indices, _) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                indices
            })
            .collect()
    }
}

impl<T: DistanceValue + Send + Sync> ParPartition<T> for BalancedBall<T> {
    fn par_new<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        data: &D,
        metric: &M,
        indices: &[usize],
        depth: usize,
    ) -> Result<Self, String> {
        Ball::par_new(data, metric, indices, depth).map(|ball| Self(ball, Vec::new()))
    }

    fn par_find_extrema<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &self,
        data: &D,
        metric: &M,
    ) -> Vec<usize> {
        self.0.par_find_extrema(data, metric)
    }

    #[allow(clippy::similar_names)]
    fn par_split_by_extrema<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &self,
        data: &D,
        metric: &M,
        extrema: &[usize],
    ) -> Vec<Vec<usize>> {
        let [l, r] = [extrema[0], extrema[1]];
        let lr = data.par_one_to_one(l, r, metric);

        let items = self.par_indices().filter(|&i| !(i == l || i == r)).collect::<Vec<_>>();

        // Find the distances from each extremum to each item.
        let l_distances = data.par_one_to_many(l, &items, metric).collect::<Vec<_>>();
        let r_distances = data.par_one_to_many(r, &items, metric).collect::<Vec<_>>();

        let child_stacks = {
            let lr = lr
                .to_f32()
                .unwrap_or_else(|| unreachable!("Cannot convert distance to f32"));

            // Find the distance from `l` to each item projected onto the line
            // connecting `l` and `r`.
            let lr_distances = {
                let mut lr_distances = l_distances
                    .into_par_iter()
                    .map(|(a, d)| {
                        (
                            a,
                            d.to_f32()
                                .unwrap_or_else(|| unreachable!("Cannot convert distance to f32")),
                        )
                    })
                    .zip(r_distances.into_par_iter().map(|(_, d)| {
                        d.to_f32()
                            .unwrap_or_else(|| unreachable!("Cannot convert distance to f32"))
                    }))
                    .map(|((a, al), ar)| {
                        let cos = ar.mul_add(-ar, lr.mul_add(lr, al.powi(2))) / (2.0 * al * lr);
                        (a, al * cos)
                    })
                    .collect::<Vec<_>>();
                lr_distances.sort_by(|(_, a), (_, b)| a.total_cmp(b));
                lr_distances
                    .into_par_iter()
                    .map(|(a, d)| {
                        (
                            a,
                            T::from_f32(d).unwrap_or_else(|| {
                                unreachable!("Cannot convert f32 to {}", core::any::type_name::<T>())
                            }),
                        )
                    })
                    .collect::<Vec<_>>()
            };

            // Half of the items will be assigned to the left child and the
            // other half to the right child.
            let mid = if lr_distances.len() % 2 == 0 {
                lr_distances.len() / 2
            } else {
                1 + lr_distances.len() / 2
            };
            let (ls, rs) = lr_distances.split_at(mid);
            let l_stack = core::iter::once((l, T::zero()))
                .chain(ls.iter().copied())
                .collect::<Vec<_>>();
            let r_stack = core::iter::once((r, T::zero()))
                .chain(rs.iter().copied())
                .collect::<Vec<_>>();

            vec![l_stack, r_stack]
        };

        child_stacks
            .into_iter()
            .map(|stack| {
                let (indices, _) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                indices
            })
            .collect()
    }
}

impl<T: DistanceValue + ToBytes<Bytes = Vec<u8>> + FromBytes<Bytes = Vec<u8>>> crate::DiskIO for BalancedBall<T> {
    fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        let ball_bytes = self.0.to_bytes()?;
        bytes.extend_from_slice(&ball_bytes.len().to_le_bytes());
        bytes.extend_from_slice(&ball_bytes);

        for c in &self.1 {
            let child_bytes = c.to_bytes()?;
            bytes.extend_from_slice(&child_bytes.len().to_le_bytes());
            bytes.extend_from_slice(&child_bytes);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut offset = 0;
        let ball_bytes = crate::utils::read_encoding(bytes, &mut offset);
        let ball = Ball::from_bytes(&ball_bytes)?;

        let mut children = Vec::new();
        while offset < bytes.len() {
            let child_bytes = crate::utils::read_encoding(bytes, &mut offset);
            let child = Self::from_bytes(&child_bytes)?;
            children.push(Box::new(child));
        }

        Ok(Self(ball, children))
    }
}

impl<T: DistanceValue + ToBytes<Bytes = Vec<u8>> + FromBytes<Bytes = Vec<u8>> + Send + Sync> crate::ParDiskIO
    for BalancedBall<T>
{
}
