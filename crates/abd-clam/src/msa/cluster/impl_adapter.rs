//! Implementation of the `Adapter` trait for the `PartialMSA` struct.

use distances::number::UInt;

use crate::{
    adapter::{Adapter, BallAdapter, ParAdapter, ParBallAdapter, ParParams, Params},
    cluster::ParCluster,
    dataset::ParDataset,
    Ball, Cluster, Dataset,
};

use super::{Alignable, GapIds, PartialMSA};

impl<I, U: UInt, D: Dataset<I, U>, S: Cluster<I, U, D>> Params<I, U, D, D, S> for GapIds {
    fn child_params(&self, children: &[S]) -> Vec<Self> {
        children.iter().map(|_| Self::default()).collect()
    }
}

impl<I: Send + Sync, U: UInt, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParParams<I, U, D, D, S> for GapIds {
    fn par_child_params(&self, children: &[S]) -> Vec<Self> {
        self.child_params(children)
    }
}

impl<I: Alignable + AsRef<str>, U: UInt, D: Dataset<I, U>, S: Cluster<I, U, D>> Adapter<I, U, D, D, S, GapIds>
    for PartialMSA<I, U, D, S>
{
    fn new_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, _: GapIds, d_in: &D) -> Self {
        let params = if children.is_empty() {
            GapIds::default()
        } else {
            // Here we assume that there are only two children.
            let l_child = children[0].2.as_ref();
            let r_child = children[1].2.as_ref();

            // We get the centers for the left and right children.
            let l_center = l_child.aligned_point(d_in, l_child.arg_center());
            let r_center = r_child.aligned_point(d_in, r_child.arg_center());
            let (x, y) = (l_center.as_ref(), r_center.as_ref());

            // We get the indices at which gaps need to be added to bring the
            // children into alignment.
            let [left, right] = distances::strings::needleman_wunsch::alignment_gaps::<U>(x, y);

            GapIds { left, right }
        };

        Self {
            source,
            children,
            gap_ids: params,
            _phantom: core::marker::PhantomData,
        }
    }

    fn post_traversal(&mut self) {}

    fn source(&self) -> &S {
        &self.source
    }

    fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    fn take_source(self) -> S {
        self.source
    }

    fn params(&self) -> &GapIds {
        &self.gap_ids
    }
}

impl<I: Alignable + AsRef<str> + Send + Sync, U: UInt, D: ParDataset<I, U>, S: ParCluster<I, U, D>>
    ParAdapter<I, U, D, D, S, GapIds> for PartialMSA<I, U, D, S>
{
    fn par_new_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, _: GapIds, d_in: &D) -> Self {
        let params = if children.is_empty() {
            GapIds::default()
        } else {
            // Here we assume that there are only two children.
            let l_child = children[0].2.as_ref();
            let r_child = children[1].2.as_ref();

            // We get the centers for the left and right children.
            let (l_center, r_center) = rayon::join(
                || l_child.aligned_point(d_in, l_child.arg_center()),
                || r_child.aligned_point(d_in, r_child.arg_center()),
            );
            let (l_center, r_center) = (l_center.as_ref(), r_center.as_ref());

            // We get the indices at which gaps need to be added to bring the
            // children into alignment.
            let [left, right] = distances::strings::needleman_wunsch::x_to_y_alignment(l_center, r_center);

            GapIds { left, right }
        };

        Self {
            source,
            children,
            gap_ids: params,
            _phantom: core::marker::PhantomData,
        }
    }

    fn par_post_traversal(&mut self) {}
}

impl<I: Alignable + AsRef<str>, U: UInt, D: Dataset<I, U>> BallAdapter<I, U, D, D, GapIds>
    for PartialMSA<I, U, D, Ball<I, U, D>>
{
    fn from_ball_tree(ball: Ball<I, U, D>, data: D) -> (Self, D) {
        let root = Self::adapt_tree_iterative(ball, None, &data);
        (root, data)
    }
}

impl<I: Alignable + AsRef<str> + Send + Sync, U: UInt, D: ParDataset<I, U>> ParBallAdapter<I, U, D, D, GapIds>
    for PartialMSA<I, U, D, Ball<I, U, D>>
{
    fn par_from_ball_tree(ball: Ball<I, U, D>, data: D) -> (Self, D) {
        let root = Self::par_adapt_tree_iterative(ball, None, &data);
        (root, data)
    }
}
