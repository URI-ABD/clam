//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use core::hash::{Hash, Hasher};

use distances::Number;

use super::PartitionCriteria;
use crate::{dataset::Dataset, utils::helpers};

pub type Ratios = [f64; 6];

/// A `Cluster` represents a collection of "similar" instances from a
/// metric-`Space`.
///
/// `Cluster`s can be unwieldy to use directly unless one has a good grasp of
/// the underlying invariants.
/// We anticipate that most users' needs will be well met by the higher-level
/// abstractions.
///
/// For most use-cases, one should chain calls to `new_root`, `build` and
/// `partition` to construct a tree on the metric space.
///
/// Clusters are named in the same way as nodes in a Huffman tree. The `root` is
/// named "1". A left child appends a "0" to the name of the parent and a right
/// child appends a "1".
///
/// For now, `Cluster` names are unique within a single tree. We plan on adding
/// tree-based prefixes which will make names unique across multiple trees.
#[derive(Debug)]
pub(crate) struct Cluster<T: Send + Sync + Copy, U: Number> {
    pub history: Vec<bool>,
    pub seed: Option<u64>,
    pub offset: usize,
    pub cardinality: usize,
    pub center: T,
    #[allow(dead_code)]
    pub radial: T,
    pub radius: U,
    pub arg_radius: usize,
    #[allow(dead_code)]
    pub lfd: f64,
    pub children: Option<Children<T, U>>,

    #[allow(dead_code)]
    pub ratios: Option<Ratios>,
}

#[derive(Debug)]
pub(crate) struct Children<T: Send + Sync + Copy, U: Number> {
    pub left: Box<Cluster<T, U>>,
    pub right: Box<Cluster<T, U>>,
    pub l_pole: T,
    pub r_pole: T,
    pub polar_distance: U,
}

impl<T: Send + Sync + Copy, U: Number> PartialEq for Cluster<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.history == other.history
    }
}

/// Two clusters are equal if they have the same name. This only holds, for
/// now, for clusters in the same tree.
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

/// `Cluster`s can be sorted based on their name. `Cluster`s are sorted by
/// non-decreasing depths and then by their names. Sorting a tree of `Cluster`s
/// will leave them in the order of a breadth-first traversal.
impl<T: Send + Sync + Copy, U: Number> Ord for Cluster<T, U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.depth() == other.depth() {
            self.offset.cmp(&other.offset)
        } else {
            self.depth().cmp(&other.depth())
        }
    }
}

/// Clusters are hashed by their names. This means that a hash is only unique
/// within a single tree.
///
/// TODO: Add a tree-based prefix to the cluster names when we need to hash
/// clusters from different trees into the same collection.
impl<T: Send + Sync + Copy, U: Number> Hash for Cluster<T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.offset, self.cardinality).hash(state)
    }
}

impl<T: Send + Sync + Copy, U: Number> std::fmt::Display for Cluster<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<T: Send + Sync + Copy, U: Number> Cluster<T, U> {
    /// Creates a new root `Cluster` for the metric space.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    pub fn new_root<D: Dataset<T, U>>(data: &D, indices: &[usize], seed: Option<u64>) -> Self {
        Cluster::new(data, seed, vec![true], 0, indices)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    /// * `indices`: The indices of instances from the `dataset` that are
    /// contained in the `Cluster`.
    /// * `name`: `BitVec` name for the `Cluster`.
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
            let n = ((indices.len() as f64).sqrt()) as usize;
            data.choose_unique(n, indices, seed)
        };

        let arg_center = data.median(&arg_samples);
        let center = data.get(arg_center);

        let center_distances = data.one_to_many(arg_center, indices);
        let (arg_radius, radius) = helpers::arg_max(&center_distances);
        let arg_radius = indices[arg_radius];
        let radial = data.get(arg_radius);

        let lfd = helpers::compute_lfd(radius, &center_distances);

        Cluster {
            history,
            seed,
            offset,
            cardinality,
            center,
            radial,
            radius,
            arg_radius,
            lfd,
            children: None,
            ratios: None,
        }
    }

    pub fn partition<D: Dataset<T, U>>(mut self, data: &mut D, criteria: &PartitionCriteria<T, U>) -> Self {
        assert_eq!(self.depth(), 0, "This method may only be called on a root cluster.");

        let mut indices = data.indices().to_vec();
        (self, indices) = self._partition(data, criteria, 0, indices);
        data.reorder(&indices);

        self
    }

    fn _partition<D: Dataset<T, U>>(
        mut self,
        data: &D,
        criteria: &PartitionCriteria<T, U>,
        offset: usize,
        mut indices: Vec<usize>,
    ) -> (Self, Vec<usize>) {
        if criteria.check(&self) {
            let ([(l_pole, l_indices), (r_pole, r_indices)], polar_distance) = self.partition_once(data, indices);

            let (l_offset, r_offset) = (offset, offset + l_indices.len());

            // TODO: Insert parallelism here
            let ((left, l_indices), (right, r_indices)) = (
                Cluster::new(data, self.seed, self.child_history(false), l_offset, &l_indices)
                    ._partition(data, criteria, l_offset, l_indices),
                Cluster::new(data, self.seed, self.child_history(true), r_offset, &r_indices)
                    ._partition(data, criteria, r_offset, r_indices),
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
        (self, indices)
    }

    fn partition_once<D: Dataset<T, U>>(&self, data: &D, indices: Vec<usize>) -> ([(T, Vec<usize>); 2], U) {
        let l_pole = self.arg_radius;
        let l_distances = data.one_to_many(l_pole, &indices);

        let (arg_r, polar_distance) = helpers::arg_max(&l_distances);
        let r_pole = indices[arg_r];
        let r_distances = data.one_to_many(r_pole, &indices);

        let (l, r) = indices
            .into_iter()
            .zip(l_distances.into_iter())
            .zip(r_distances.into_iter())
            .filter(|&((i, _), _)| i != l_pole && i != r_pole)
            .partition::<Vec<_>, _>(|&((_, l), r)| l <= r);

        let l_indices = l
            .into_iter()
            .map(|((i, _), _)| i)
            .chain([l_pole].into_iter())
            .collect::<Vec<_>>();
        let r_indices = r
            .into_iter()
            .map(|((i, _), _)| i)
            .chain([r_pole].into_iter())
            .collect::<Vec<_>>();

        let (l_pole, r_pole) = (data.get(l_pole), data.get(r_pole));
        if l_indices.len() < r_indices.len() {
            ([(r_pole, r_indices), (l_pole, l_indices)], polar_distance)
        } else {
            ([(l_pole, l_indices), (r_pole, r_indices)], polar_distance)
        }
    }

    fn child_history(&self, right: bool) -> Vec<bool> {
        let mut history = self.history.clone();
        history.push(right);
        history
    }

    /// Computes and sets the `Ratios` for all `Cluster`s in the tree. These
    /// ratios are used for selecting `Graph`s for anomaly detection and other
    /// applications of CLAM.
    ///
    /// This method may only be called on a root cluster after calling the `build`
    /// and `partition` methods.
    ///
    /// # Arguments
    ///
    /// * `normalized`: Whether to normalize each ratio to a [0, 1] range based
    /// on the distribution of values for all `Cluster`s in the tree.
    ///
    /// # Panics:
    ///
    /// * If called on a non-root `Cluster`, i.e. a `Cluster` with depth > 0.
    /// * If called before `build` and `partition`.
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

    #[inline(always)]
    #[allow(dead_code)]
    fn next_ema(&self, ratio: f64, parent_ema: f64) -> f64 {
        // TODO: Consider getting `alpha` from user. Perhaps via env vars?
        let alpha = 2. / 11.;
        alpha * ratio + (1. - alpha) * parent_ema
    }

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

    pub fn indices<'a, D: Dataset<T, U>>(&'a self, data: &'a D) -> &[usize] {
        &data.indices()[self.offset..(self.offset + self.cardinality)]
    }

    // /// Returns a Vector of indices that corresponds to a depth-first traversal of
    // /// the children of a given cluster. This function is distinguished from `indices`
    // /// in that it creates a `Vec` that has all of the indices for a given cluster
    // /// hierarchy instead of returning a reference to a given cluster's indices.
    // ///
    // pub fn leaf_indices(&self) -> Vec<usize> {
    //     match &self.index {
    //         Index::Empty => match &self.children {
    //             Some(([(_, left), (_, right)], _)) => left
    //                 .leaf_indices()
    //                 .iter()
    //                 .chain(right.leaf_indices().iter())
    //                 .copied()
    //                 .collect(),

    //             // TODO: Cleanup this error message
    //             None => panic!("Structural invariant invalidated. Node with no contents and no children"),
    //         },
    //         Index::Indices(indices) => indices.clone(),
    //         Index::Offset(_) => {
    //             panic!("Cannot get leaf indices once tree has been reordered!");
    //         }
    //     }
    // }

    /// The `history` of the `Cluster` as a bool vector.
    #[allow(dead_code)]
    pub fn history(&self) -> &[bool] {
        &self.history
    }

    /// The `name` of the `Cluster` as a hex-String.
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
                let s = u8::from_str_radix(&s, 2).unwrap();
                format!("{s:01x}")
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Whether the `Cluster` is the root of the tree.
    ///
    /// The root `Cluster` has a depth of 0.
    #[allow(dead_code)]
    pub fn is_root(&self) -> bool {
        self.depth() == 0
    }

    /// The number of parent-child hops from the root `Cluster` to this one.
    pub fn depth(&self) -> usize {
        self.history.len() - 1
    }

    /// Whether the `Cluster` contains only one instance or only identical
    /// instances.
    pub fn is_singleton(&self) -> bool {
        self.radius == U::zero()
    }

    /// Whether this cluster has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// A 2-slice of references to the left and right child `Cluster`s.
    pub fn children(&self) -> Option<[&Self; 2]> {
        self.children.as_ref().map(|v| [v.left.as_ref(), v.right.as_ref()])
    }

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
    /// * exponential moving average of child-cardinality / parent-cardinality.
    /// * exponential moving average of child-radius / parent-radius.
    /// * exponential moving average of child-lfd / parent-lfd.
    ///
    /// This method may only be called after calling `with_ratios` on the root.
    ///
    /// # Panics:
    ///
    /// * If called before calling `with_ratios` on the root.
    #[allow(dead_code)]
    pub fn ratios(&self) -> Ratios {
        self.ratios
            .expect("Please call `with_ratios` before using this method.")
    }

    /// Whether this `Cluster` is an ancestor of the `other` `Cluster`.
    #[allow(dead_code)]
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.history.iter().zip(other.history.iter()).all(|(&l, &r)| l == r)
    }

    /// Whether this `Cluster` is an descendant of the `other` `Cluster`.
    #[allow(dead_code)]
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// A Vec of references to all `Cluster`s in the subtree of this `Cluster`,
    /// including this `Cluster`.
    ///
    /// Note: Calling this function is potentially expensive.
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
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    /// Distance from the `center` to the given indexed instance.
    #[allow(dead_code)]
    pub fn distance_to_indexed_instance<D: Dataset<T, U>>(&self, data: &D, index: usize) -> U {
        data.metric()(data.get(index), self.center)
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
        let children = self
            .children
            .as_ref()
            .expect("This method may only be called on non-leaf clusters.");
        let ql = data.metric()(query, children.l_pole);
        let qr = data.metric()(query, children.r_pole);

        let swap = ql < qr;
        let (ql, qr) = if swap { (qr, ql) } else { (ql, qr) };

        if (ql + qr) * (ql - qr) <= U::from(2) * children.polar_distance * radius {
            vec![&children.left, &children.right]
        } else if swap {
            vec![&children.left]
        } else {
            vec![&children.right]
        }
    }
}

#[cfg(test)]
mod tests {
    use distances::vectors::euclidean;

    use crate::{
        cluster::Tree,
        dataset::{Dataset, VecVec},
    };

    use super::*;

    #[test]
    fn test_cluster() {
        let data: Vec<&[f32]> = vec![&[0., 0., 0.], &[1., 1., 1.], &[2., 2., 2.], &[3., 3., 3.]];
        let name = "test".to_string();
        let mut data = VecVec::new(data, euclidean::<f32, f32>, name, false);
        let indices = data.indices().to_vec();
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
        let cluster = Cluster::new_root(&data, &indices, Some(42)).partition(&mut data, &partition_criteria);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality, 4);
        assert_eq!(cluster.subtree().len(), 7);
        assert!(cluster.radius > 0.);

        assert_eq!(format!("{cluster}"), "1");

        let [left, right] = cluster.children().unwrap();
        assert_eq!(format!("{left}"), "2");
        assert_eq!(format!("{right}"), "3");

        for child in [left, right] {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality, 2);
            assert_eq!(child.subtree().len(), 3);
        }
    }

    #[test]
    fn test_leaf_indices() {
        let data: Vec<&[f32]> = vec![&[10.], &[1.], &[-5.], &[8.], &[3.], &[2.], &[0.5], &[0.]];
        let name = "test".to_string();
        let data = VecVec::new(data, euclidean::<f32, f32>, name, false);
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

        let tree = Tree::new(data, Some(42)).partition(&partition_criteria);

        let mut leaf_indices = tree.root().indices(tree.data()).to_vec();
        leaf_indices.sort();

        assert_eq!(leaf_indices, tree.data().indices());
    }

    #[test]
    fn test_end_to_end_reordering() {
        let data: Vec<&[f32]> = vec![&[10.], &[1.], &[-5.], &[8.], &[3.], &[2.], &[0.5], &[0.]];
        let name = "test".to_string();
        let data = VecVec::new(data, euclidean::<f32, f32>, name, false);
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
        let data = data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
        let name = "test".to_string();
        let mut data = VecVec::<_, f32>::new(data, euclidean, name, false);
        let indices = data.indices().to_vec();
        let partition_criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        let root = Cluster::new_root(&data, &indices, Some(seed)).partition(&mut data, &partition_criteria);

        for c in root.subtree() {
            assert!(c.cardinality > 0, "Cardinality must be positive.");
            assert!(c.radius >= 0., "Radius must be non-negative.");
            assert!(c.lfd > 0., "LFD must be positive.");

            let radius = data.metric()(c.center, c.radial);
            assert_eq!(
                c.radius, radius,
                "Radius must be equal to the distance to the farthest instance."
            );
        }
    }
}
