//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use std::hash::Hash;
use std::hash::Hasher;

use bitvec::prelude::*;

use super::cluster_criteria::PartitionCriteria;
use super::dataset::Dataset;
use super::number::Number;
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
pub struct Cluster<'a, T: Number, U: Number, D: Dataset<T, U>> {
    t: std::marker::PhantomData<T>,
    data: &'a D,
    cardinality: usize,
    contents: Contents<'a, T, U, D>,
    history: BitVec,
    arg_center: Option<usize>,
    arg_radius: Option<usize>,
    radius: Option<U>,
    lfd: Option<f64>,
    ratios: Option<Ratios>,
    seed: Option<u64>,
}

#[derive(Debug)]
#[allow(clippy::type_complexity)]
enum Contents<'a, T: Number, U: Number, D: Dataset<T, U>> {
    // indices are only held at leaf clusters. This helps reduce the memory
    // footprint of the tree.
    Indices(Vec<usize>),
    // ([(left_pole, left_child), (right_pole, right_child)], polar_distance)
    Children(([(usize, Box<Cluster<'a, T, U, D>>); 2], U)),
}

impl<'a, T: Number, U: Number, D: Dataset<T, U>> PartialEq for Cluster<'a, T, U, D> {
    fn eq(&self, other: &Self) -> bool {
        self.history == other.history
    }
}

/// Two clusters are equal if they have the same name. This only holds, for
/// now, for clusters in the same tree.
impl<'a, T: Number, U: Number, D: Dataset<T, U>> Eq for Cluster<'a, T, U, D> {}

impl<'a, T: Number, U: Number, D: Dataset<T, U>> PartialOrd for Cluster<'a, T, U, D> {
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
impl<'a, T: Number, U: Number, D: Dataset<T, U>> Ord for Cluster<'a, T, U, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Clusters are hashed by their names. This means that a hash is only unique
/// within a single tree.
///
/// TODO: Add a tree-based prefix to the cluster names when we need to hash
/// clusters from different trees into the same collection.
impl<'a, T: Number, U: Number, D: Dataset<T, U>> Hash for Cluster<'a, T, U, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl<'a, T: Number, U: Number, D: Dataset<T, U>> std::fmt::Display for Cluster<'a, T, U, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<'a, T: Number, U: Number, D: Dataset<T, U>> Cluster<'a, T, U, D> {
    /// Creates a new root `Cluster` for the metric space.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    pub fn new_root(dataset: &'a D) -> Self {
        let name = bitvec![1];
        let indices = dataset.indices();
        Cluster::new(dataset, indices, name)
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    /// * `indices`: The indices of instances from the `dataset` that are
    /// contained in the `Cluster`.
    /// * `name`: `BitVec` name for the `Cluster`.
    pub fn new(data: &'a D, indices: Vec<usize>, history: BitVec) -> Self {
        Cluster {
            t: Default::default(),
            data,
            cardinality: indices.len(),
            contents: Contents::Indices(indices),
            history,
            arg_center: None,
            arg_radius: None,
            radius: None,
            lfd: None,
            ratios: None,
            seed: None,
        }
    }

    /// Computes and sets internal cluster properties including:
    /// - `arg_samples`
    /// - `arg_center`
    /// - `arg_radius`
    /// - `radius`
    /// - `lfd` (local fractal dimension)
    ///
    /// This method must be called before calling `partition` and before
    /// using the getter methods for those internal properties.
    pub fn build(mut self) -> Self {
        let indices = match &self.contents {
            Contents::Indices(indices) => indices,
            Contents::Children(_) => panic!("`build` can only be called once per cluster."),
        };

        // TODO: Explore with different values for the threshold e.g. 10, 100, 1000, etc.
        let arg_samples = if self.cardinality < 100 {
            indices.clone()
        } else {
            let n = ((indices.len() as f64).sqrt()) as usize;
            self.data.choose_unique(n, indices, self.seed)
        };

        let arg_center = self.data.median(&arg_samples);

        let center_distances = self.data.one_to_many(arg_center, indices);
        let (arg_radius, radius) = helpers::arg_max(&center_distances);
        let arg_radius = indices[arg_radius];

        self.arg_center = Some(arg_center);
        self.arg_radius = Some(arg_radius);
        self.radius = Some(radius);
        self.lfd = Some(helpers::compute_lfd(radius, &center_distances));

        self
    }

    /// Returns two new `Cluster`s that are the left and right children of this
    /// `Cluster`.
    fn partition_once(&mut self) -> ([(usize, Self); 2], U) {
        let indices = match &self.contents {
            Contents::Indices(indices) => indices,
            Contents::Children(_) => panic!("`build` can only be called once per cluster."),
        };

        let left_pole = self.arg_radius();
        let left_distances = self.data.one_to_many(left_pole, indices);

        let (arg_right, polar_distance) = helpers::arg_max(&left_distances);
        let right_pole = indices[arg_right];
        let right_distances = self.data.one_to_many(right_pole, indices);

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

        let left = Cluster::new(self.data, left_indices, left_name).build();
        let right = Cluster::new(self.data, right_indices, right_name).build();

        ([(left_pole, left), (right_pole, right)], polar_distance)
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
    pub fn partition(mut self, criteria: &PartitionCriteria<T, U, D>, recursive: bool) -> Self {
        if criteria.check(&self) {
            let ([(left_pole, left), (right_pole, right)], polar_distance) = self.partition_once();

            let (left, right) = if recursive {
                (
                    left.partition(criteria, recursive),
                    right.partition(criteria, recursive),
                )
            } else {
                (left, right)
            };

            self.contents = Contents::Children((
                [(left_pole, Box::new(left)), (right_pole, Box::new(right))],
                polar_distance,
            ));
        }
        self
    }

    pub fn par_partition(mut self, criteria: &PartitionCriteria<T, U, D>, recursive: bool) -> Self {
        if criteria.check(&self) {
            let ([(left_pole, left), (right_pole, right)], polar_distance) = self.partition_once();

            let (left, right) = if recursive {
                rayon::join(
                    || left.par_partition(criteria, recursive),
                    || right.par_partition(criteria, recursive),
                )
            } else {
                (left, right)
            };

            self.contents = Contents::Children((
                [(left_pole, Box::new(left)), (right_pole, Box::new(right))],
                polar_distance,
            ));
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
    #[allow(unused_mut, unused_variables)]
    pub fn with_ratios(mut self, normalized: bool) -> Self {
        todo!()
        // if !self.is_root() {
        //     panic!("This method may only be set from the root cluster.")
        // }
        // if self.is_leaf() {
        //     panic!("Please `build` and `partition` the tree before setting cluster ratios.")
        // }

        // match &self.contents {
        //     Contents::Indices(_) => panic!("Should not be here ..."),
        //     Contents::Children(([(l, left), (r, right)], lr)) => {
        //         let left = Box::new(left.set_child_parent_ratios([1.; 6]));
        //         let right = Box::new(right.set_child_parent_ratios([1.; 6]));
        //         self.contents = Contents::Children(([(*l, left), (*r, right)], *lr));
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

        // match &self.contents {
        //     Contents::Indices(_) => (),
        //     Contents::Children(([(l, left), (r, right)], lr)) => {
        //         let left = Box::new(left.set_child_parent_ratios([1.; 6]));
        //         let right = Box::new(right.set_child_parent_ratios([1.; 6]));
        //         self.contents = Contents::Children(([(*l, left), (*r, right)], *lr));
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

        // match self.contents {
        //     Contents::Indices(_) => (),
        //     Contents::Children(([(_, mut left), (_, mut right)], _)) => {
        //         left.set_normalized_ratios(means, sds);
        //         right.set_normalized_ratios(means, sds);
        //     },
        // };
    }

    /// The underlying dataset
    pub fn dataset(&'a self) -> &dyn Dataset<T, U> {
        self.data
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
    pub fn indices(&self) -> Vec<usize> {
        match &self.contents {
            Contents::Indices(indices) => indices.clone(),
            Contents::Children(([(_, left), (_, right)], _)) => {
                left.indices().into_iter().chain(right.indices().into_iter()).collect()
            }
        }
    }

    /// The `history` of the `Cluster` as a bool vector.
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
            .expect("Please call `build` on this cluster before using this method.")
    }

    /// The index of the instance that is farthest from the `center`.
    pub fn arg_radius(&self) -> usize {
        self.arg_radius
            .expect("Please call `build` on this cluster before using this method.")
    }

    /// The distance between the `center` and the instance farthest from the
    /// `center`.
    pub fn radius(&self) -> U {
        self.radius
            .expect("Please call `build` on this cluster before using this method.")
    }

    /// Whether the `Cluster` contains only one instance or only identical
    /// instances.
    pub fn is_singleton(&self) -> bool {
        self.radius() == U::zero()
    }

    /// The local fractal dimension of the `Cluster` at the length scales of the
    /// `radius` and half that `radius`.
    pub fn lfd(&self) -> f64 {
        self.lfd
            .expect("Please call `build` on this cluster before using this method.")
    }

    pub fn polar_distance(&self) -> Option<U> {
        match &self.contents {
            Contents::Indices(_) => None,
            Contents::Children((_, lr)) => Some(*lr),
        }
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
    pub fn ratios(&self) -> Ratios {
        self.ratios
            .expect("Please call `with_ratios` before using this method.")
    }

    /// A 2-slice of references to the left and right child `Cluster`s.
    pub fn children(&self) -> Option<[&Self; 2]> {
        match &self.contents {
            Contents::Indices(_) => None,
            Contents::Children(([(_, left), (_, right)], _)) => Some([left.as_ref(), right.as_ref()]),
        }
    }

    /// Whether this cluster has no children.
    pub fn is_leaf(&self) -> bool {
        matches!(&self.contents, Contents::Indices(_))
    }

    /// Whether this `Cluster` is an ancestor of the `other` `Cluster`.
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.history.iter().zip(other.history.iter()).all(|(l, r)| *l == *r)
    }

    /// Whether this `Cluster` is an descendant of the `other` `Cluster`.
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// A Vec of references to all `Cluster`s in the subtree of this `Cluster`,
    /// including this `Cluster`.
    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];
        match &self.contents {
            Contents::Indices(_) => subtree,
            Contents::Children(([(_, left), (_, right)], _)) => subtree
                .into_iter()
                .chain(left.subtree().into_iter())
                .chain(right.subtree().into_iter())
                .collect(),
        }
    }

    /// The number of descendants of this `Cluster`, excluding itself.
    pub fn num_descendants(&self) -> usize {
        self.subtree().len() - 1
    }

    /// The maximum depth of any leaf in the subtree of this `Cluster`.
    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    /// Distance from the `center` to the given indexed instance.
    pub fn distance_to_indexed_instance(&self, index: usize) -> U {
        self.data.one_to_one(index, self.arg_center())
    }

    /// Distance from the `center` to the given instance.
    pub fn distance_to_instance(&self, instance: &[T]) -> U {
        self.data.query_to_one(instance, self.arg_center())
    }

    /// Distance from the `center` of this `Cluster` to the center of the
    /// `other` `Cluster`.
    pub fn distance_to_other(&self, other: &Self) -> U {
        self.distance_to_indexed_instance(other.arg_center())
    }

    /// Assuming that this `Cluster` overlaps with with query ball, we return
    /// only those children that also overlap with the query ball
    pub fn overlapping_children(&self, query: &[T], radius: U) -> Vec<&Self> {
        let (l, left, r, right, lr) = match &self.contents {
            Contents::Indices(_) => panic!("Can only be called on non-leaf clusters."),
            Contents::Children(([(l, left), (r, right)], lr)) => (*l, left.as_ref(), *r, right.as_ref(), *lr),
        };
        let ql = self.data.query_to_one(query, l);
        let qr = self.data.query_to_one(query, r);

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
}

#[cfg(test)]
mod tests {
    use crate::core::cluster::Cluster;
    use crate::core::cluster_criteria::PartitionCriteria;
    use crate::core::dataset::VecVec;
    use crate::utils::distances;

    #[test]
    fn test_cluster() {
        let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
        let metric = distances::euclidean::<f32, f32>;
        let name = "test".to_string();
        let space = VecVec::new(data, metric, name, false);
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);

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
}
