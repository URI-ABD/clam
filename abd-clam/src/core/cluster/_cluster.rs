//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use core::hash::{Hash, Hasher};

use distances::Number;

use crate::{utils, Dataset, PartitionCriteria, PartitionCriterion};

/// Ratios are used for anomaly detection and related applications.
pub type Ratios = [f64; 6];

/// A `Cluster` represents a collection of "similar" instances from a metric-`Space`.
///
/// `Cluster`s can be unwieldy to use directly unless one has a good grasp of
/// the underlying invariants. We anticipate that most users' needs will be well
/// met by the higher-level abstractions, e.g. `Tree`, `Graph`, `CAKES`, etc.
///
/// For now, `Cluster` names are unique within a single tree. We plan on adding
/// tree-based prefixes which will make names unique across multiple trees.
#[derive(Debug)]
pub struct Cluster<T: Send + Sync + Copy, U: Number> {
    /// The `Cluster`'s history in the tree.
    pub history: Vec<bool>,
    /// The seed used in the random number generator for this `Cluster`.
    pub seed: Option<u64>,
    /// The offset of the indices of the `Cluster`'s instances in the dataset.
    pub offset: usize,
    /// The number of instances in the `Cluster`.
    pub cardinality: usize,
    /// The geometric mean of the `Cluster`.
    pub center: T,
    /// The index of the `center` instance in the dataset.
    pub arg_center: usize,
    /// The instance furthest from the `center` of the `Cluster`.
    pub radial: T,
    /// The index of the `radial` instance in the dataset.
    pub arg_radial: usize,
    /// The distance from the `center` to the `radial` instance.
    pub radius: U,
    /// The local fractal dimension of the `Cluster`.
    #[allow(dead_code)]
    pub lfd: f64,
    /// The children of the `Cluster`.
    pub children: Option<Children<T, U>>,
    /// The six `Cluster` ratios used for anomaly detection and related applications.
    #[allow(dead_code)]
    pub ratios: Option<Ratios>,
}

/// The children of a `Cluster`.
#[derive(Debug)]
pub struct Children<T: Send + Sync + Copy, U: Number> {
    /// The left child of the `Cluster`.
    pub left: Box<Cluster<T, U>>,
    /// The right child of the `Cluster`.
    pub right: Box<Cluster<T, U>>,
    /// The left pole of the `Cluster` (i.e. the instance used to identify
    /// instances for the left child).
    pub l_pole: T,
    /// The right pole of the `Cluster` (i.e. the instance used to identify
    /// instances for the right child).
    pub r_pole: T,
    /// The distance from the `l_pole` to the `r_pole` instance.
    pub polar_distance: U,
}

impl<T: Send + Sync + Copy, U: Number> PartialEq for Cluster<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.history == other.history
    }
}

impl<T: Send + Sync + Copy, U: Number> Eq for Cluster<T, U> {}

impl<T: Send + Sync + Copy, U: Number> PartialOrd for Cluster<T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.depth() == other.depth() {
            self.offset.partial_cmp(&other.offset)
        } else {
            self.depth().partial_cmp(&other.depth())
        }
    }
}

impl<T: Send + Sync + Copy, U: Number> Ord for Cluster<T, U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.depth() == other.depth() {
            self.offset.cmp(&other.offset)
        } else {
            self.depth().cmp(&other.depth())
        }
    }
}

impl<T: Send + Sync + Copy, U: Number> Hash for Cluster<T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.offset, self.cardinality).hash(state);
    }
}

impl<T: Send + Sync + Copy, U: Number> std::fmt::Display for Cluster<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<T: Send + Sync + Copy, U: Number> Cluster<T, U> {
    /// Creates a new root `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `data`: on which to create the `Cluster`.
    /// * `indices`: The indices of instances from the `dataset` that are contained in the `Cluster`.
    /// * `seed`: The seed used in the random number generator for this `Cluster`.
    pub fn new_root<D: Dataset<T, U>>(data: &D, indices: &[usize], seed: Option<u64>) -> Self {
        Self::new(data, seed, vec![true], 0, indices)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `data`: on which to create the `Cluster`.
    /// * `seed`: The seed used in the random number generator for this `Cluster`.
    /// * `history`: The `Cluster`'s history in the tree.
    /// * `offset`: The offset of the indices of the `Cluster`'s instances in the dataset.
    /// * `indices`: The indices of instances from the `dataset` that are contained in the `Cluster`.
    pub fn new<D: Dataset<T, U>>(
        data: &D,
        seed: Option<u64>,
        history: Vec<bool>,
        offset: usize,
        indices: &[usize],
    ) -> Self {
        let cardinality = indices.len();

        // TODO: Explore with different values for the threshold e.g. 10, 100, 1000, etc.
        let arg_samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = (indices.len().as_f64().sqrt()) as usize;
            data.choose_unique(n, indices, seed)
        };

        let Some(arg_center) = data.median(&arg_samples) else { unreachable!("The cluster should have at least one instance.") };

        let center = data.get(arg_center);

        let center_distances = data.one_to_many(arg_center, indices);
        let Some((arg_radial, radius)) = utils::arg_max(&center_distances) else { unreachable!("The cluster should have at least one instance.") };
        let arg_radial = indices[arg_radial];
        let radial = data.get(arg_radial);

        let lfd = utils::compute_lfd(radius, &center_distances);

        Self {
            history,
            seed,
            offset,
            cardinality,
            center,
            arg_center,
            radial,
            arg_radial,
            radius,
            lfd,
            children: None,
            ratios: None,
        }
    }

    /// Partitions the `Cluster` into two children if the `Cluster` meets the
    /// given `PartitionCriteria`.
    ///
    /// This method should only be called on a root `Cluster`. It is user error
    /// to call this method on a non-root `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `data`: The `Dataset` for the `Cluster`.
    /// * `criteria`: The `PartitionCriteria` to use for partitioning.
    ///
    /// # Returns
    ///
    /// * The `Cluster` on which the method was called after partitioning
    /// recursively until the `PartitionCriteria` is no longer met on any of the
    /// leaf `Cluster`s.
    pub fn partition<D: Dataset<T, U>>(mut self, data: &mut D, criteria: &PartitionCriteria<T, U>) -> Self {
        let mut indices = data.indices().to_vec();
        (self, indices) = self._partition(data, criteria, indices);
        data.reorder(&indices);

        self
    }

    /// Recursive helper function for `partition`.
    fn _partition<D: Dataset<T, U>>(
        mut self,
        data: &D,
        criteria: &PartitionCriteria<T, U>,
        mut indices: Vec<usize>,
    ) -> (Self, Vec<usize>) {
        if criteria.check(&self) {
            let ([(l_pole, l_indices), (r_pole, r_indices)], polar_distance) = self.partition_once(data, indices);

            let r_offset = self.offset + l_indices.len();

            let ((left, l_indices), (right, r_indices)) = rayon::join(
                || {
                    Self::new(data, self.seed, self.child_history(false), self.offset, &l_indices)
                        ._partition(data, criteria, l_indices)
                },
                || {
                    Self::new(data, self.seed, self.child_history(true), r_offset, &r_indices)
                        ._partition(data, criteria, r_indices)
                },
            );

            let (left, right) = (Box::new(left), Box::new(right));

            indices = l_indices.into_iter().chain(r_indices.into_iter()).collect::<Vec<_>>();

            self.children = Some(Children {
                left,
                right,
                l_pole,
                r_pole,
                polar_distance,
            });
        }

        // reset the indices to center and radial indices for data reordering
        let (arg_center, _) = utils::pos_val(&indices, self.arg_center)
            .unwrap_or_else(|| unreachable!("We know the center is in the indices."));
        self.arg_center = self.offset + arg_center;

        let (arg_radial, _) = utils::pos_val(&indices, self.arg_radial)
            .unwrap_or_else(|| unreachable!("We know the radial is in the indices."));
        self.arg_radial = self.offset + arg_radial;

        (self, indices)
    }

    /// Partitions the `Cluster` into two children once.
    fn partition_once<D: Dataset<T, U>>(&self, data: &D, indices: Vec<usize>) -> ([(T, Vec<usize>); 2], U) {
        let l_distances = data.query_to_many(self.radial, &indices);

        let Some((arg_r, polar_distance)) = utils::arg_max(&l_distances) else { unreachable!("The cluster should have at least one instance.") };
        let r_pole = data.get(indices[arg_r]);
        let r_distances = data.query_to_many(r_pole, &indices);

        let (l_indices, r_indices) = indices
            .into_iter()
            .zip(l_distances.into_iter())
            .zip(r_distances.into_iter())
            .partition::<Vec<_>, _>(|&((_, l), r)| l <= r);

        let l_indices = Self::drop_distances(l_indices);
        let r_indices = Self::drop_distances(r_indices);

        if l_indices.len() < r_indices.len() {
            ([(r_pole, r_indices), (self.radial, l_indices)], polar_distance)
        } else {
            ([(self.radial, l_indices), (r_pole, r_indices)], polar_distance)
        }
    }

    /// Drops the distances from a vector, returning only the indices.
    fn drop_distances(indices: Vec<((usize, U), U)>) -> Vec<usize> {
        indices.into_iter().map(|((i, _), _)| i).collect()
    }

    /// The `history` of the child `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `right`: Whether the child `Cluster` is the right child.
    fn child_history(&self, right: bool) -> Vec<bool> {
        let mut history = self.history.clone();
        history.push(right);
        history
    }

    /// Sets the `Cluster` ratios for anomaly detection and related applications.
    ///
    /// This method may only be called on the root `Cluster`. It is user error
    /// to call this method on a non-root `Cluster`.
    ///
    /// This method should be called after calling `partition` on the root
    /// `Cluster`. It is user error to call this method before calling
    /// `partition` on the root `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `normalized`: Whether to normalize the ratios. We use Gaussian error
    /// functions to normalize the ratios, which is a common practice in
    /// anomaly detection.
    #[allow(unused_mut, unused_variables, dead_code)]
    pub fn with_ratios(mut self, normalized: bool) -> Self {
        todo!()
        // if !self.is_root() {
        //     panic!("This method may only be set from the root cluster.")
        // }
        // if self.is_leaf() {
        //     panic!("Please `build` and `partition` the tree before setting cluster ratios.")
        // }

        // match &self.index {
        //     Index::Indices(_) => panic!("Should not be here ..."),
        //     Index::Children(([(l, left), (r, right)], lr)) => {
        //         let left = Box::new(left.set_child_parent_ratios([1.; 6]));
        //         let right = Box::new(right.set_child_parent_ratios([1.; 6]));
        //         self.index = Index::Children(([(*l, left), (*r, right)], *lr));
        //     },
        // };
        // self.ratios = Some([1.; 6]);

        // if normalized {
        //     let ratios: Vec<_> = self.subtree().iter().flat_map(|c| c.ratios()).collect();
        //     let ratios: Vec<Vec<_>> = (0..6)
        //         .map(|s| ratios.iter().skip(s).step_by(6).cloned().collect())
        //         .collect();
        //     let means: [f64; 6] = ratios
        //         .iter()
        //         .map(|values| helpers::mean(values))
        //         .collect::<Vec<_>>()
        //         .try_into()
        //         .unwrap();
        //     let sds: [f64; 6] = ratios
        //         .iter()
        //         .zip(means.iter())
        //         .map(|(values, &mean)| 1e-8 + helpers::sd(values, mean))
        //         .collect::<Vec<_>>()
        //         .try_into()
        //         .unwrap();

        //     self.set_normalized_ratios(means, sds);
        // }

        // self
    }

    /// Sets the chile-parent `Cluster` ratios for anomaly detection and related
    /// applications.
    ///
    /// # Arguments
    ///
    /// * `parent_ratios`: The ratios for the parent `Cluster`.
    #[allow(unused_mut, unused_variables, dead_code)]
    fn set_child_parent_ratios(mut self, parent_ratios: Ratios) -> Self {
        todo!()
        // let [pc, pr, pl, pc_, pr_, pl_] = parent_ratios;

        // let c = (self.cardinality as f64) / pc;
        // let r = self.radius().as_f64() / pr;
        // let l = self.lfd() / pl;

        // let c_ = self.next_ema(c, pc_);
        // let r_ = self.next_ema(r, pr_);
        // let l_ = self.next_ema(l, pl_);

        // let ratios = [c, r, l, c_, r_, l_];
        // self.ratios = Some(ratios);

        // match &self.index {
        //     Index::Indices(_) => (),
        //     Index::Children(([(l, left), (r, right)], lr)) => {
        //         let left = Box::new(left.set_child_parent_ratios([1.; 6]));
        //         let right = Box::new(right.set_child_parent_ratios([1.; 6]));
        //         self.index = Index::Children(([(*l, left), (*r, right)], *lr));
        //     },
        // };

        // self
    }

    /// Normalizes the `Cluster` ratios for anomaly detection and related
    /// applications.
    ///
    /// # Arguments
    ///
    /// * `means`: The means of the `Cluster` ratios.
    /// * `sds`: The standard deviations of the `Cluster` ratios.
    #[allow(unused_mut, unused_variables, dead_code)]
    fn set_normalized_ratios(&mut self, means: Ratios, sds: Ratios) {
        todo!()
        // let ratios: Vec<_> = self
        //     .ratios
        //     .unwrap()
        //     .into_iter()
        //     .zip(means.into_iter())
        //     .zip(sds.into_iter())
        //     .map(|((value, mean), std)| (value - mean) / (std * 2_f64.sqrt()))
        //     .map(libm::erf)
        //     .map(|v| (1. + v) / 2.)
        //     .collect();
        // self.ratios = Some(ratios.try_into().unwrap());

        // match self.index {
        //     Index::Indices(_) => (),
        //     Index::Children(([(_, mut left), (_, mut right)], _)) => {
        //         left.set_normalized_ratios(means, sds);
        //         right.set_normalized_ratios(means, sds);
        //     },
        // };
    }

    /// The indices of the `Cluster`'s instances in the dataset.
    pub fn indices<'a, D: Dataset<T, U>>(&'a self, data: &'a D) -> &[usize] {
        &data.indices()[self.offset..(self.offset + self.cardinality)]
    }

    /// The `history` of the `Cluster` as a bool vector.
    ///
    /// The root `Cluster` has a `history` of length 1, being `vec![true]`. The
    /// history of a left child is the history of its parent with a `false`
    /// appended. The history of a right child is the history of its parent with
    /// a `true` appended.
    ///
    /// The `history` of a `Cluster` is used to identify the `Cluster` in the
    /// tree, and to compute the `Cluster`'s `name`.
    #[allow(dead_code)]
    pub fn history(&self) -> &[bool] {
        &self.history
    }

    /// The `name` of the `Cluster` as a hex-String.
    ///
    /// This is a human-readable representation of the `Cluster`'s `history`.
    /// It may be used to store the `Cluster` in a database, or to identify the
    /// `Cluster` in a visualization.
    #[allow(clippy::many_single_char_names)]
    pub fn name(&self) -> String {
        let d = self.history.len();
        let padding = if d % 4 == 0 { 0 } else { 4 - d % 4 };
        let bin_name = (0..padding)
            .map(|_| "0")
            .chain(self.history.iter().map(|&b| if b { "1" } else { "0" }))
            .collect::<Vec<_>>();
        bin_name
            .chunks_exact(4)
            .map(|s| {
                let [a, b, c, d] = [s[0], s[1], s[2], s[3]];
                let s = format!("{a}{b}{c}{d}");
                let Ok(s) = u8::from_str_radix(&s, 2) else { unreachable!("We know the characters used are only \"0\" and \"1\".") };
                format!("{s:01x}")
            })
            .collect()
    }

    /// Whether the `Cluster` is the root of the tree.
    ///
    /// The root `Cluster` has a depth of 0.
    #[allow(dead_code)]
    pub fn is_root(&self) -> bool {
        self.depth() == 0
    }

    /// The depth of the `Cluster` in the tree.
    ///
    /// The root `Cluster` has a depth of 0. The depth of a child is the depth
    /// of its parent plus 1.
    pub fn depth(&self) -> usize {
        self.history.len() - 1
    }

    /// Whether the `Cluster` contains only one instance or only identical
    /// instances.
    pub fn is_singleton(&self) -> bool {
        self.radius == U::zero()
    }

    /// Whether this cluster has no children.
    pub const fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// A 2-slice of references to the left and right child `Cluster`s.
    pub fn children(&self) -> Option<[&Self; 2]> {
        self.children.as_ref().map(|v| [v.left.as_ref(), v.right.as_ref()])
    }

    /// The distance between the poles of the `Cluster`.
    #[allow(dead_code)]
    pub fn polar_distance(&self) -> Option<U> {
        self.children.as_ref().map(|v| v.polar_distance)
    }

    /// The six `Cluster` ratios used for anomaly detection and related
    /// applications.
    ///
    /// These ratios are:
    ///
    /// * child-cardinality / parent-cardinality.
    /// * child-radius / parent-radius.
    /// * child-lfd / parent-lfd.
    /// * EMA of child-cardinality / parent-cardinality.
    /// * EMA of child-radius / parent-radius.
    /// * EMA of child-lfd / parent-lfd.
    ///
    /// This method may only be called after calling `with_ratios` on the root.
    /// It is user error to call this method before calling `with_ratios` on
    /// the root.
    #[allow(dead_code)]
    pub fn ratios(&self) -> Ratios {
        self.ratios.unwrap_or([0.; 6])
    }

    /// Whether this `Cluster` is an ancestor of the `other` `Cluster`.
    #[allow(dead_code)]
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.history.as_slice() == &other.history[..self.history.len()]
    }

    /// Whether this `Cluster` is an descendant of the `other` `Cluster`.
    #[allow(dead_code)]
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// A Vec of references to all `Cluster`s in the subtree of this `Cluster`,
    /// including this `Cluster`.
    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];

        // Two scenarios: Either we have children or not
        match self.children() {
            Some([left, right]) => subtree
                .into_iter()
                .chain(left.subtree().into_iter())
                .chain(right.subtree().into_iter())
                .collect(),

            None => subtree,
        }
    }

    /// The maximum depth of any leaf in the subtree of this `Cluster`.
    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(Self::depth).max().map_or_else(
            || unreachable!("The subtree of a Cluster should have at least one element, i.e. the Cluster itself."),
            |depth| depth,
        )
    }

    /// Distance from the `center` to the given instance.
    pub fn distance_to_instance<D: Dataset<T, U>>(&self, data: &D, instance: T) -> U {
        data.metric()(instance, self.center)
    }

    /// Distance from the `center` of this `Cluster` to the center of the
    /// `other` `Cluster`.
    #[allow(dead_code)]
    pub fn distance_to_other<D: Dataset<T, U>>(&self, data: &D, other: &Self) -> U {
        data.metric()(self.center, other.center)
    }

    /// Assuming that this `Cluster` overlaps with with query ball, we return
    /// only those children that also overlap with the query ball
    pub fn overlapping_children<D: Dataset<T, U>>(&self, data: &D, query: T, radius: U) -> Vec<&Self> {
        self.children.as_ref().map_or_else(
            Vec::new,
            |Children {
                 left,
                 right,
                 l_pole,
                 r_pole,
                 polar_distance,
             }| {
                let ql = data.metric()(query, *l_pole);
                let qr = data.metric()(query, *r_pole);

                let swap = ql < qr;
                let (ql, qr) = if swap { (qr, ql) } else { (ql, qr) };

                if (ql + qr) * (ql - qr) <= U::from(2) * (*polar_distance) * radius {
                    vec![left.as_ref(), right.as_ref()]
                } else if swap {
                    vec![left.as_ref()]
                } else {
                    vec![right.as_ref()]
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use core::f32::EPSILON;

    use distances::vectors::euclidean;

    use crate::{
        Tree, {Dataset, VecDataset},
    };

    use super::*;

    #[test]
    fn test_cluster() {
        let data: Vec<&[f32]> = vec![&[0., 0., 0.], &[1., 1., 1.], &[2., 2., 2.], &[3., 3., 3.]];
        let name = "test".to_string();
        let mut data = VecDataset::new(name, data, euclidean::<f32, f32>, false);
        let indices = data.indices().to_vec();
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
        let root = Cluster::new_root(&data, &indices, Some(42)).partition(&mut data, &partition_criteria);

        assert!(!root.is_leaf());
        assert!(root.children().is_some());

        assert_eq!(root.depth(), 0);
        assert_eq!(root.cardinality, 4);
        assert_eq!(root.subtree().len(), 7);
        assert!(root.radius > 0.);

        assert_eq!(format!("{root}"), "1");

        let Some([left, right]) = root.children() else { unreachable!("The root cluster has children.") };
        assert_eq!(format!("{left}"), "2");
        assert_eq!(format!("{right}"), "3");

        for child in [left, right] {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality, 2);
            assert_eq!(child.subtree().len(), 3);
        }

        let subtree = root.subtree();
        assert_eq!(
            subtree.len(),
            7,
            "The subtree of the root cluster should have 7 elements but had {}.",
            subtree.len()
        );
        for c in root.subtree() {
            let center_d = data.query_to_one(c.center, c.arg_center);
            assert!(
                center_d <= f32::EPSILON,
                "Center must be the closest instance to itself. {c} had center {:?} and argcenter {} in data {:?}",
                c.center,
                c.arg_center,
                data.data
            );

            let radial_d = data.query_to_one(c.radial, c.arg_radial);
            assert!(
                radial_d <= f32::EPSILON,
                "Radial must be the closest instance to itself. {c} had radial {:?} but argradial {} in data {:?}",
                c.radial,
                c.arg_radial,
                data.data
            );
        }
    }

    #[test]
    fn test_leaf_indices() {
        let data: Vec<&[f32]> = vec![&[10.], &[1.], &[-5.], &[8.], &[3.], &[2.], &[0.5], &[0.]];
        let name = "test".to_string();
        let data = VecDataset::new(name, data, euclidean::<f32, f32>, false);
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

        let tree = Tree::new(data, Some(42)).partition(&partition_criteria);

        let mut leaf_indices = tree.root().indices(tree.data()).to_vec();
        leaf_indices.sort_unstable();

        assert_eq!(leaf_indices, tree.data().indices());
    }

    #[test]
    fn test_end_to_end_reordering() {
        let data: Vec<&[f32]> = vec![&[10.], &[1.], &[-5.], &[8.], &[3.], &[2.], &[0.5], &[0.]];
        let name = "test".to_string();
        let data = VecDataset::new(name, data, euclidean::<f32, f32>, false);
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

        let tree = Tree::new(data, Some(42)).partition(&partition_criteria);

        // Assert that the root's indices actually cover the whole dataset.
        assert_eq!(tree.data().cardinality(), tree.indices().len());

        // Assert that the tree's indices have been reordered in depth-first order
        assert_eq!((0..tree.cardinality()).collect::<Vec<usize>>(), tree.indices());
    }

    #[test]
    fn cluster() {
        let (dimensionality, min_val, max_val) = (10, -1., 1.);
        let seed = 42;

        let data = symagen::random_data::random_f32(10_000, dimensionality, min_val, max_val, seed);
        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let name = "test".to_string();
        let mut data = VecDataset::<_, f32>::new(name, data, euclidean, false);
        let indices = data.indices().to_vec();
        let partition_criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        let root = Cluster::new_root(&data, &indices, Some(seed)).partition(&mut data, &partition_criteria);

        for c in root.subtree() {
            assert!(c.cardinality > 0, "Cardinality must be positive.");
            assert!(c.radius >= 0., "Radius must be non-negative.");
            assert!(c.lfd > 0., "LFD must be positive.");

            let radius = data.metric()(c.center, c.radial);
            assert!(
                (c.radius - radius).abs() < EPSILON,
                "Radius must be equal to the distance to the farthest instance."
            );

            let center_d = data.query_to_one(c.center, c.arg_center);
            assert!(
                center_d <= f32::EPSILON,
                "Center must be the closest instance to itself. {c} had {center_d}"
            );

            let radial_d = data.query_to_one(c.radial, c.arg_radial);
            assert!(
                radial_d <= f32::EPSILON,
                "Radial must be the closest instance to itself. {c} had {radial_d}"
            );
        }
    }
}
