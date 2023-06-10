//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use std::hash::Hash;
use std::hash::Hasher;

use bitvec::prelude::*;

use super::PartitionCriteria;
use crate::dataset::Dataset;
use crate::number::Number;
use crate::utils::helpers;

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
pub(crate) struct Cluster<T: Number, U: Number, D: Dataset<T, U>> {
    _t: std::marker::PhantomData<T>,
    _d: std::marker::PhantomData<D>,
    cardinality: usize,
    history: BitVec,
    arg_center: usize,
    arg_radius: usize,
    radius: U,
    #[allow(dead_code)]
    lfd: f64,
    #[allow(dead_code)]
    ratios: Option<Ratios>,
    seed: Option<u64>,

    // TODO: Simplify this type by using a `Children` struct
    #[allow(clippy::type_complexity)]
    children: Option<([(usize, Box<Cluster<T, U, D>>); 2], U)>,
    index: Index,
}

#[derive(Debug)]
enum Index {
    // Leaf nodes only (Direct access)
    Indices(Vec<usize>),

    // All nodes after reordering (Direct access)
    Offset(usize),

    // Root nodes only (Indirect access through traversal )
    Empty,
}

impl<T: Number, U: Number, D: Dataset<T, U>> PartialEq for Cluster<T, U, D> {
    fn eq(&self, other: &Self) -> bool {
        self.history == other.history
    }
}

/// Two clusters are equal if they have the same name. This only holds, for
/// now, for clusters in the same tree.
impl<T: Number, U: Number, D: Dataset<T, U>> Eq for Cluster<T, U, D> {}

impl<T: Number, U: Number, D: Dataset<T, U>> PartialOrd for Cluster<T, U, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.depth() == other.depth() {
            self.history.partial_cmp(&other.history)
        } else {
            self.depth().partial_cmp(&other.depth())
        }
    }
}

/// `Cluster`s can be sorted based on their name. `Cluster`s are sorted by
/// non-decreasing depths and then by their names. Sorting a tree of `Cluster`s
/// will leave them in the order of a breadth-first traversal.
impl<T: Number, U: Number, D: Dataset<T, U>> Ord for Cluster<T, U, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Clusters are hashed by their names. This means that a hash is only unique
/// within a single tree.
///
/// TODO: Add a tree-based prefix to the cluster names when we need to hash
/// clusters from different trees into the same collection.
impl<T: Number, U: Number, D: Dataset<T, U>> Hash for Cluster<T, U, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl<T: Number, U: Number, D: Dataset<T, U>> std::fmt::Display for Cluster<T, U, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<T: Number, U: Number, D: Dataset<T, U>> Cluster<T, U, D> {
    /// Creates a new root `Cluster` for the metric space.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    pub fn new_root(data: &D, indices: &[usize], seed: Option<u64>) -> Self {
        let name = bitvec![1];
        Cluster::new(data, indices, name, seed)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    /// * `indices`: The indices of instances from the `dataset` that are
    /// contained in the `Cluster`.
    /// * `name`: `BitVec` name for the `Cluster`.
    pub fn new(data: &D, indices: &[usize], history: BitVec, seed: Option<u64>) -> Self {
        let cardinality = indices.len();

        // TODO: Explore with different values for the threshold e.g. 10, 100, 1000, etc.
        let arg_samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            let n = ((indices.len() as f64).sqrt()) as usize;
            data.choose_unique(n, indices, seed)
        };

        let arg_center = data.median(&arg_samples);

        let center_distances = data.one_to_many(arg_center, indices);
        let (arg_radius, radius) = helpers::arg_max(&center_distances);
        let arg_radius = indices[arg_radius];

        Cluster {
            _t: Default::default(),
            _d: Default::default(),
            cardinality,
            history,
            arg_center,
            arg_radius,
            radius,
            lfd: helpers::compute_lfd(radius, &center_distances),
            ratios: None,
            seed,
            children: None,
            index: Index::Indices(indices.to_vec()),
        }
    }

    /// Returns two new `Cluster`s that are the left and right children of this
    /// `Cluster`.
    fn partition_once(&self, data: &D) -> ([(usize, Vec<usize>, BitVec); 2], U) {
        let indices = match &self.index {
            Index::Indices(indices) => indices,
            _ => panic!("`build` can only be called once per cluster."),
        };

        let left_pole = self.arg_radius();
        let left_distances = data.one_to_many(left_pole, indices);

        let (arg_right, polar_distance) = helpers::arg_max(&left_distances);
        let right_pole = indices[arg_right];
        let right_distances = data.one_to_many(right_pole, indices);

        let (left, right) = indices
            .iter()
            .zip(left_distances.into_iter())
            .zip(right_distances.into_iter())
            .filter(|&((&i, _), _)| i != left_pole && i != right_pole)
            .partition::<Vec<_>, _>(|&((_, l), r)| l <= r);

        let left_indices = left
            .into_iter()
            .map(|((&i, _), _)| i)
            .chain([left_pole].into_iter())
            .collect::<Vec<_>>();
        let right_indices = right
            .into_iter()
            .map(|((&i, _), _)| i)
            .chain([right_pole].into_iter())
            .collect::<Vec<_>>();

        let (left_pole, left_indices, right_pole, right_indices) = if left_indices.len() < right_indices.len() {
            (right_pole, right_indices, left_pole, left_indices)
        } else {
            (left_pole, left_indices, right_pole, right_indices)
        };

        let left_name = {
            let mut name = self.history.clone();
            name.push(false);
            name
        };
        let right_name = {
            let mut name = self.history.clone();
            name.push(true);
            name
        };

        (
            [
                (left_pole, left_indices, left_name),
                (right_pole, right_indices, right_name),
            ],
            polar_distance,
        )
    }

    /// Partitions the `Cluster` based on the given criteria. If the `Cluster`
    /// can be partitioned, it will gain a pair of left and right child
    /// `Cluster`s. If called with the `recursive` flag, this will build the
    /// tree down to leaf `Cluster`s, i.e. `Cluster`s that can not be
    /// partitioned based on the given criteria.
    ///
    /// This method should be called after calling `build` and before calling
    /// the getter methods for children.
    ///
    /// # Arguments
    ///
    /// * `partition_criteria`: The rules by which to determine whether the
    /// cluster can be partitioned.
    /// * `recursive`: Whether to build the tree down to leaves using the same
    /// `partition_criteria`.
    ///
    /// # Panics:
    ///
    /// * If called before calling `build`.
    pub fn partition(mut self, data: &D, criteria: &PartitionCriteria<T, U, D>, recursive: bool) -> Self {
        if criteria.check(&self) {
            let ([(left_pole, left_indices, left_name), (right_pole, right_indices, right_name)], polar_distance) =
                self.partition_once(data);

            let (left, right) = (
                Cluster::new(data, &left_indices, left_name, self.seed),
                Cluster::new(data, &right_indices, right_name, self.seed),
            );

            let (left, right) = if recursive {
                (
                    left.partition(data, criteria, recursive),
                    right.partition(data, criteria, recursive),
                )
            } else {
                (left, right)
            };

            self.children = Some((
                [(left_pole, Box::new(left)), (right_pole, Box::new(right))],
                polar_distance,
            ));
            self.index = Index::Empty;
        }
        self
    }

    pub fn par_partition(mut self, data: &D, criteria: &PartitionCriteria<T, U, D>, recursive: bool) -> Self {
        if criteria.check(&self) {
            let ([(left_pole, left_indices, left_name), (right_pole, right_indices, right_name)], polar_distance) =
                self.partition_once(data);

            let (left, right) = rayon::join(
                || Cluster::new(data, &left_indices, left_name, self.seed),
                || Cluster::new(data, &right_indices, right_name, self.seed),
            );

            let (left, right) = if recursive {
                rayon::join(
                    || left.par_partition(data, criteria, recursive),
                    || right.par_partition(data, criteria, recursive),
                )
            } else {
                (left, right)
            };

            self.children = Some((
                [(left_pole, Box::new(left)), (right_pole, Box::new(right))],
                polar_distance,
            ));
            self.index = Index::Empty;
        }
        self
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

    /// The number of instances in this `Cluster`.
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns the indices of the instances contained in this `Cluster`.
    ///
    /// Indices are only stored at leaf `Cluster`s. Calling this method on a
    /// non-leaf `Cluster` will have to perform a tree traversal, returning the
    /// indices in depth-first order.
    pub fn indices<'a>(&'a self, data: &'a D) -> &[usize] {
        match &self.index {
            Index::Indices(indices) => indices,
            Index::Offset(o) => {
                let start = *o;
                &data.indices()[start..start + self.cardinality]
            }
            Index::Empty => panic!("Cannot call indices from parent clusters"),
        }
    }

    /// Returns a Vector of indices that corresponds to a depth-first traversal of
    /// the children of a given cluster. This function is distingushed from `indices`
    /// in that it creates a `Vec` that has all of the incides for a given cluster
    /// hierarchy instead of returning a reference to a given cluster's indices.
    ///
    pub fn leaf_indices(&self) -> Vec<usize> {
        match &self.index {
            Index::Empty => match &self.children {
                Some(([(_, left), (_, right)], _)) => left
                    .leaf_indices()
                    .iter()
                    .chain(right.leaf_indices().iter())
                    .copied()
                    .collect(),

                // TODO: Cleanup this error message
                None => panic!("Structural invariant invalidated. Node with no contents and no children"),
            },
            Index::Indices(indices) => indices.clone(),
            Index::Offset(_) => {
                panic!("Cannot get leaf indices once tree has been reordered!");
            }
        }
    }

    /// The `history` of the `Cluster` as a bool vector.
    #[allow(dead_code)]
    pub fn history(&self) -> Vec<bool> {
        self.history.iter().map(|v| *v).collect()
    }

    /// The `name` of the `Cluster` as a hex-String.
    pub fn name(&self) -> String {
        let d = self.history.len();
        let padding = if d % 4 == 0 { 0 } else { 4 - d % 4 };
        let bin_name = (0..padding)
            .map(|_| "0")
            .chain(self.history.iter().map(|b| if *b { "1" } else { "0" }))
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

    /// The index of the instance at the center, i.e. the geometric median, of
    /// the `Cluster`.
    ///
    /// For `Cluster`s with a large `cardinality`, this is an approximation.
    ///
    /// TODO: Analyze the level of approximation for this. It's probably a
    /// sqrt(3) approximation based on some work in computational geometry.
    pub fn arg_center(&self) -> usize {
        self.arg_center
    }

    /// The index of the instance that is farthest from the `center`.
    pub fn arg_radius(&self) -> usize {
        self.arg_radius
    }

    /// The distance between the `center` and the instance farthest from the
    /// `center`.
    pub fn radius(&self) -> U {
        self.radius
    }

    /// Whether the `Cluster` contains only one instance or only identical
    /// instances.
    pub fn is_singleton(&self) -> bool {
        self.radius() == U::zero()
    }

    /// The local fractal dimension of the `Cluster` at the length scales of the
    /// `radius` and half that `radius`.
    #[allow(dead_code)]
    pub fn lfd(&self) -> f64 {
        self.lfd
    }

    #[allow(dead_code)]
    pub fn polar_distance(&self) -> Option<U> {
        self.children.as_ref().map(|(_, lr)| *lr)
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

    /// A 2-slice of references to the left and right child `Cluster`s.
    pub fn children(&self) -> Option<[&Self; 2]> {
        self.children
            .as_ref()
            .map(|([(_, left), (_, right)], _)| [left.as_ref(), right.as_ref()])
    }

    /// Whether this cluster has no children.
    pub fn is_leaf(&self) -> bool {
        matches!(&self.index, Index::Indices(_))
    }

    /// Whether this `Cluster` is an ancestor of the `other` `Cluster`.
    #[allow(dead_code)]
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.history.iter().zip(other.history.iter()).all(|(l, r)| *l == *r)
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
        match &self.children {
            Some(([(_, left), (_, right)], _)) => subtree
                .into_iter()
                .chain(left.subtree().into_iter())
                .chain(right.subtree().into_iter())
                .collect(),

            None => subtree,
        }
    }

    /// The number of descendants of this `Cluster`, excluding itself.
    #[allow(dead_code)]
    pub fn num_descendants(&self) -> usize {
        self.subtree().len() - 1
    }

    /// The maximum depth of any leaf in the subtree of this `Cluster`.
    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    /// Distance from the `center` to the given indexed instance.
    #[allow(dead_code)]
    pub fn distance_to_indexed_instance(&self, data: &D, index: usize) -> U {
        data.one_to_one(index, self.arg_center())
    }

    /// Distance from the `center` to the given instance.
    pub fn distance_to_instance(&self, data: &D, instance: &[T]) -> U {
        data.query_to_one(instance, self.arg_center())
    }

    /// Distance from the `center` of this `Cluster` to the center of the
    /// `other` `Cluster`.
    #[allow(dead_code)]
    pub fn distance_to_other(&self, data: &D, other: &Self) -> U {
        self.distance_to_indexed_instance(data, other.arg_center())
    }

    /// Assuming that this `Cluster` overlaps with with query ball, we return
    /// only those children that also overlap with the query ball
    pub fn overlapping_children(&self, data: &D, query: &[T], radius: U) -> Vec<&Self> {
        let (l, left, r, right, lr) = match &self.children {
            None => panic!("Can only be called on non-leaf clusters."),
            Some(([(l, left), (r, right)], lr)) => (*l, left.as_ref(), *r, right.as_ref(), *lr),
        };
        let ql = data.query_to_one(query, l);
        let qr = data.query_to_one(query, r);

        let swap = ql < qr;
        let (ql, qr) = if swap { (qr, ql) } else { (ql, qr) };

        if (ql + qr) * (ql - qr) <= U::from(2).unwrap() * lr * radius {
            vec![left, right]
        } else if swap {
            vec![left]
        } else {
            vec![right]
        }
    }

    #[allow(dead_code)]
    pub fn depth_first_reorder(&mut self, data: &D) {
        if self.depth() != 0 {
            panic!("Cannot call this method except from the root.")
        }

        self.dfr(data, 0);
    }

    pub fn dfr(&mut self, data: &D, offset: usize) {
        self.index = Index::Offset(offset);

        // TODO: Cleanup
        self.arg_center = data.get_reordered_index(self.arg_center);
        self.arg_radius = data.get_reordered_index(self.arg_radius);

        if let Some(([(_, left), (_, right)], _)) = self.children.as_mut() {
            left.dfr(data, offset);
            right.dfr(data, offset + left.cardinality);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cluster::Tree;
    use crate::dataset::{Dataset, VecVec};
    use crate::distances;

    use super::*;

    #[test]
    fn test_cluster() {
        let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
        let metric = distances::f32::euclidean;
        let name = "test".to_string();
        let data = VecVec::new(data, metric, name, false);
        let indices = data.indices().to_vec();
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
        let cluster = Cluster::new_root(&data, &indices, Some(42)).partition(&data, &partition_criteria, true);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 4);
        assert_eq!(cluster.num_descendants(), 6);
        assert!(cluster.radius() > 0.);

        assert_eq!(format!("{cluster}"), "1");

        let [left, right] = cluster.children().unwrap();
        assert_eq!(format!("{left}"), "2");
        assert_eq!(format!("{right}"), "3");

        for child in [left, right] {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 2);
            assert_eq!(child.num_descendants(), 2);
        }
    }

    #[test]
    fn test_leaf_indices() {
        let data = vec![
            vec![10.],
            vec![1.],
            vec![-5.],
            vec![8.],
            vec![3.],
            vec![2.],
            vec![0.5],
            vec![0.],
        ];
        let metric = distances::f32::euclidean;
        let name = "test".to_string();
        let data = VecVec::new(data, metric, name, false);
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

        let tree = Tree::new(data, Some(42)).partition(&partition_criteria, true);

        let mut leaf_indices = tree.root().leaf_indices();
        leaf_indices.sort();

        assert_eq!(leaf_indices, tree.data().indices());
    }

    mod reordering {
        use super::*;

        #[test]
        fn test_end_to_end_reordering() {
            let data = vec![
                vec![10.],
                vec![1.],
                vec![-5.],
                vec![8.],
                vec![3.],
                vec![2.],
                vec![0.5],
                vec![0.],
            ];
            let metric = distances::f32::euclidean;
            let name = "test".to_string();
            let data = VecVec::new(data, metric, name, false);
            let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

            let tree = Tree::new(data, Some(42))
                .partition(&partition_criteria, true)
                .depth_first_reorder();

            // Assert that the root's indices actually cover the whole dataset.
            assert_eq!(tree.data().cardinality(), tree.indices().len());

            // Assert that the tree's indices have been reordered in depth-first order
            assert_eq!((0..tree.cardinality()).collect::<Vec<usize>>(), tree.indices());
        }

        #[test]
        fn test_tree_transformation_before_after_reordering() {
            // Test that assures the pre and post reorder Index assignment works
            // as expected.
            let data = vec![
                vec![10.],
                vec![1.],
                vec![-5.],
                vec![8.],
                vec![3.],
                vec![2.],
                vec![0.5],
                vec![0.],
            ];
            let metric = distances::f32::euclidean;
            let name = "test".to_string();
            let data = VecVec::new(data, metric, name, false);
            let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

            let tree = Tree::new(data, Some(42)).partition(&partition_criteria, true);

            assert!(matches!(tree.root().index, Index::Empty));

            let tree = tree.depth_first_reorder();

            assert!(matches!(tree.root().index, Index::Offset(0)));

            // Assert that the tree's indices have been reordered in depth-first order
            assert_eq!((0..tree.cardinality()).collect::<Vec<usize>>(), tree.indices());
        }
    }
}
