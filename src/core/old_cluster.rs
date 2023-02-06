//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use std::hash::Hash;
use std::hash::Hasher;

use bitvec::prelude::*;
// use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::helpers;
// use crate::utils::reports;

const SUB_SAMPLE_LIMIT: usize = 100;

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
#[derive(Debug, Clone)]
pub struct Cluster<'a, T: Number> {
    space: &'a dyn Space<'a, T>,
    cardinality: usize,
    // indices are only held at leaf clusters. This helps reduce the memory
    // footprint of the tree.
    indices: Option<Vec<usize>>,
    name: BitVec,
    arg_center: Option<usize>,
    arg_radius: Option<usize>,
    radius: Option<f64>,
    lfd: Option<f64>,
    left_child: Option<Box<Cluster<'a, T>>>,
    right_child: Option<Box<Cluster<'a, T>>>,
    ratios: Option<Ratios>,
}

impl<'a, T: Number> PartialEq for Cluster<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// Two clusters are equal if they have the same name. This only holds, for
/// now, for clusters in the same tree.
impl<'a, T: Number> Eq for Cluster<'a, T> {}

impl<'a, T: Number> PartialOrd for Cluster<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.depth() == other.depth() {
            self.name.partial_cmp(&other.name)
        } else {
            self.depth().partial_cmp(&other.depth())
        }
    }
}

/// `Cluster`s can be sorted based on their name. `Cluster`s are sorted by
/// non-decreasing depths and then by their names. Sorting a tree of `Cluster`s
/// will leave them in the order of a breadth-first traversal.
impl<'a, T: Number> Ord for Cluster<'a, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Clusters are hashed by their names. This means that a hash is only unique
/// within a single tree.
///
/// TODO: Add a tree-based prefix to the cluster names when we need to hash
/// clusters from different trees into the same collection.
impl<'a, T: Number> Hash for Cluster<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl<'a, T: Number> std::fmt::Display for Cluster<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name_str())
    }
}

impl<'a, T: Number> Cluster<'a, T> {
    /// Creates a new root `Cluster` for the metric space.
    ///
    /// # Arguments
    ///
    /// * `space`: metric-space on which to create the `Cluster`.
    pub fn new_root(space: &'a dyn Space<'a, T>) -> Self {
        let name = bitvec![1];
        let indices = space.data().indices();
        Cluster::new(space, indices, name)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `space`: metric-space on which to create the `Cluster`.
    /// * `indices`: The indices of instances from the `Dataset` that are
    /// contained in the `Cluster`.
    /// * `name`: `BitVec` name for the `Cluster`.
    pub fn new(space: &'a dyn Space<'a, T>, indices: Vec<usize>, name: BitVec) -> Self {
        Cluster {
            space,
            cardinality: indices.len(),
            indices: Some(indices),
            name,
            arg_center: None,
            arg_radius: None,
            radius: None,
            lfd: None,
            left_child: None,
            right_child: None,
            ratios: None,
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
        let indices = self.indices.clone().unwrap();

        let arg_samples = if indices.len() < SUB_SAMPLE_LIMIT {
            indices.clone()
        } else {
            let n = ((indices.len() as f64).sqrt()) as usize;
            self.space.choose_unique(n, &indices)
        };

        let sample_distances = self
            .space
            .pairwise(&arg_samples)
            .into_iter()
            .map(|v| v.into_iter().sum::<f64>())
            .collect::<Vec<_>>();

        let arg_center = arg_samples[helpers::arg_min(&sample_distances).0];

        let center_distances = self.space.one_to_many(arg_center, &indices);
        let (arg_radius, radius) = helpers::arg_max(&center_distances);
        let arg_radius = indices[arg_radius];

        let lfd = if radius == 0. {
            1.
        } else {
            let half_count = center_distances.into_iter().filter(|&d| d <= (radius / 2.)).count();
            if half_count > 0 {
                (indices.len().as_f64() / half_count.as_f64()).log2()
            } else {
                1.
            }
        };

        self.arg_center = Some(arg_center);
        self.arg_radius = Some(arg_radius);
        self.radius = Some(radius);
        self.lfd = Some(lfd);

        self
    }

    /// Returns two new `Cluster`s that are the left and right children of this
    /// `Cluster`.
    fn partition_once(&self) -> [Self; 2] {
        let indices = self.indices.clone().unwrap();

        let left_pole = self.arg_radius();
        let remaining_indices: Vec<usize> = indices.iter().filter(|&&i| i != left_pole).cloned().collect();
        let left_distances = self.space.one_to_many(left_pole, &remaining_indices);

        let arg_right = helpers::arg_max(&left_distances).0;
        let right_pole = remaining_indices[arg_right];

        let (left_indices, right_indices) = if remaining_indices.len() > 1 {
            let left_distances = left_distances
                .into_iter()
                .enumerate()
                .filter(|(i, _)| *i != arg_right)
                .map(|(_, d)| d);
            let remaining_indices: Vec<usize> = remaining_indices
                .iter()
                .filter(|&&i| i != right_pole)
                .cloned()
                .collect();
            let right_distances = self.space.one_to_many(right_pole, &remaining_indices);

            let (left_indices, right_indices): (Vec<_>, Vec<_>) = remaining_indices
                .into_iter()
                .zip(left_distances.zip(right_distances.into_iter()))
                .partition(|(_, (l, r))| *l <= *r);

            let (mut left_indices, _): (Vec<usize>, Vec<_>) = left_indices.into_iter().unzip();
            left_indices.push(left_pole);

            let (mut right_indices, _): (Vec<usize>, Vec<_>) = right_indices.into_iter().unzip();
            right_indices.push(right_pole);

            if left_indices.len() < right_indices.len() {
                (right_indices, left_indices)
            } else {
                (left_indices, right_indices)
            }
        } else {
            let left_indices: Vec<usize> = indices.iter().filter(|&&i| i != right_pole).cloned().collect();
            (left_indices, vec![right_pole])
        };

        let left_name = {
            let mut name = self.name.clone();
            name.push(false);
            name
        };
        let right_name = {
            let mut name = self.name.clone();
            name.push(true);
            name
        };

        let left = Cluster::new(self.space, left_indices, left_name).build();
        let right = Cluster::new(self.space, right_indices, right_name).build();

        [left, right]
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
    pub fn partition(mut self, partition_criteria: &crate::PartitionCriteria<T>, recursive: bool) -> Self {
        if partition_criteria.check(&self) {
            let [left, right] = self.partition_once();

            let (left, right) = if recursive {
                // (
                //     left.partition(partition_criteria, recursive),
                //     right.partition(partition_criteria, recursive),
                // )
                rayon::join(
                    || left.partition(partition_criteria, recursive),
                    || right.partition(partition_criteria, recursive),
                )
            } else {
                (left, right)
            };

            self.indices = None;
            self.left_child = Some(Box::new(left));
            self.right_child = Some(Box::new(right));
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
    pub fn with_ratios(mut self, normalized: bool) -> Self {
        if !self.is_root() {
            panic!("This method may only be set from the root cluster.")
        }
        if self.is_leaf() {
            panic!("Please `build` and `partition` the tree before setting cluster ratios.")
        }

        self.ratios = Some([1.; 6]);
        self.left_child = Some(Box::new(self.left_child.unwrap().set_child_parent_ratios([1.; 6])));
        self.right_child = Some(Box::new(self.right_child.unwrap().set_child_parent_ratios([1.; 6])));

        if normalized {
            let ratios: Vec<_> = self.subtree().iter().flat_map(|c| c.ratios()).collect();
            let ratios: Vec<Vec<_>> = (0..6)
                .map(|s| ratios.iter().skip(s).step_by(6).cloned().collect())
                .collect();
            let means: [f64; 6] = ratios
                .iter()
                .map(|values| helpers::mean(values))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let sds: [f64; 6] = ratios
                .iter()
                .zip(means.iter())
                .map(|(values, &mean)| 1e-8 + helpers::sd(values, mean))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            self.set_normalized_ratios(means, sds)
        } else {
            self
        }
    }

    #[inline]
    fn next_ema(&self, ratio: f64, parent_ema: f64) -> f64 {
        // TODO: Consider getting `alpha` from user. Perhaps via env vars?
        let alpha = 2. / 11.;
        alpha * ratio + (1. - alpha) * parent_ema
    }

    fn set_child_parent_ratios(mut self, parent_ratios: Ratios) -> Self {
        let [pc, pr, pl, pc_, pr_, pl_] = parent_ratios;

        let c = (self.cardinality as f64) / pc;
        let r = self.radius().as_f64() / pr;
        let l = self.lfd() / pl;

        let c_ = self.next_ema(c, pc_);
        let r_ = self.next_ema(r, pr_);
        let l_ = self.next_ema(l, pl_);

        let ratios = [c, r, l, c_, r_, l_];
        self.ratios = Some(ratios);

        if !self.is_leaf() {
            self.left_child = Some(Box::new(self.left_child.unwrap().set_child_parent_ratios(ratios)));
            self.right_child = Some(Box::new(self.right_child.unwrap().set_child_parent_ratios(ratios)));
        }

        self
    }

    fn set_normalized_ratios(mut self, means: Ratios, sds: Ratios) -> Self {
        let ratios: Vec<_> = self
            .ratios
            .unwrap()
            .into_iter()
            .zip(means.into_iter())
            .zip(sds.into_iter())
            .map(|((value, mean), std)| (value - mean) / (std * 2_f64.sqrt()))
            .map(libm::erf)
            .map(|v| (1. + v) / 2.)
            .collect();
        self.ratios = Some(ratios.try_into().unwrap());

        if !self.is_leaf() {
            self.left_child = Some(Box::new(self.left_child.unwrap().set_normalized_ratios(means, sds)));
            self.right_child = Some(Box::new(self.right_child.unwrap().set_normalized_ratios(means, sds)));
        }
        self
    }

    // pub fn report_cluster(&self) -> reports::ClusterReport {
    //     reports::ClusterReport {
    //         name: self.name_str(),
    //         cardinality: self.cardinality,
    //         indices: self.indices.clone(),
    //         arg_center: self.arg_center,
    //         arg_radius: self.arg_radius,
    //         radius: self.radius.map(|v| v.as_f64()),
    //         lfd: self.lfd,
    //         ratios: self.ratios,
    //     }
    // }

    // pub fn report_tree(&self, build_time: f64) -> (reports::TreeReport, Vec<reports::ClusterReport>) {
    //     let tree_report = reports::TreeReport {
    //         data_name: self.space.data().name(),
    //         cardinality: self.space.data().cardinality(),
    //         dimensionality: self.space.data().dimensionality(),
    //         metric_name: self.space.metric().name(),
    //         root_name: self.name_str(),
    //         max_depth: self.max_leaf_depth(),
    //         build_time,
    //     };

    //     // let cluster_reports = self.subtree().into_par_iter().map(|c| c.report_cluster()).collect();
    //     let cluster_reports = self.subtree().into_iter().map(|c| c.report_cluster()).collect();

    //     (tree_report, cluster_reports)
    // }

    /// A reference to the underlying metric space.
    pub fn space(&self) -> &dyn Space<'a, T> {
        self.space
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
        match &self.indices {
            Some(indices) => indices.clone(),
            None => self
                .left_child()
                .indices()
                .into_iter()
                .chain(self.right_child().indices().into_iter())
                .collect(),
        }
    }

    /// The `name` of the `Cluster` as a binary vector.
    ///
    /// The `name` of the `Cluster` is uniquely determined by its position in
    /// the tree.
    pub fn name(&self) -> &BitVec {
        &self.name
    }

    /// The `name` of the `Cluster` as a String of 1s and 0s.
    pub fn name_str(&self) -> String {
        let name_str: Vec<_> = self.name.iter().map(|b| if *b { "1" } else { "0" }).collect();
        name_str.join("")
    }

    /// Whether the `Cluster` is the root of the tree.
    ///
    /// The root `Cluster` has a depth of 0.
    pub fn is_root(&self) -> bool {
        self.depth() == 0
    }

    /// The number of parent-child hops from the root `Cluster` to this one.
    pub fn depth(&self) -> usize {
        self.name.len() - 1
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

    /// The instance at the center, i.e. the geometric median, of the `Cluster`.
    ///
    /// For `Cluster`s with a large `cardinality`, this is an approximation.
    pub fn center(&self) -> &'a [T] {
        self.space.data().get(self.arg_center())
    }

    /// The index of the instance that is farthest from the `center`.
    pub fn arg_radius(&self) -> usize {
        self.arg_radius
            .expect("Please call `build` on this cluster before using this method.")
    }

    /// The distance between the `center` and the instance farthest from the
    /// `center`.
    pub fn radius(&self) -> f64 {
        self.radius
            .expect("Please call `build` on this cluster before using this method.")
    }

    /// Whether the `Cluster` contains only one instance or only identical
    /// instances.
    pub fn is_singleton(&self) -> bool {
        self.radius() == 0.
    }

    /// The local fractal dimension of the `Cluster` at the length scales of the
    /// `radius` and half that `radius`.
    pub fn lfd(&self) -> f64 {
        self.lfd
            .expect("Please call `build` on this cluster before using this method.")
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

    /// A reference to the left child `Cluster`.
    pub fn left_child(&self) -> &Self {
        self.left_child
            .as_ref()
            .expect("This cluster is a leaf and has no children.")
    }

    /// A reference to the right child `Cluster`.
    pub fn right_child(&self) -> &Self {
        self.right_child
            .as_ref()
            .expect("This cluster is a leaf and has no children.")
    }

    /// A 2-slice of references to the left and right child `Cluster`s.
    pub fn children(&self) -> [&Self; 2] {
        [self.left_child(), self.right_child()]
    }

    /// Whether this cluster has no children.
    pub fn is_leaf(&self) -> bool {
        self.left_child.is_none()
    }

    /// Whether this `Cluster` is an ancestor of the `other` `Cluster`.
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.name.iter().zip(other.name.iter()).all(|(l, r)| *l == *r)
    }

    /// Whether this `Cluster` is an descendant of the `other` `Cluster`.
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// A Vec of references to all `Cluster`s in the subtree of this `Cluster`,
    /// including this `Cluster`.
    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];
        if self.is_leaf() {
            subtree
        } else {
            subtree
                .into_iter()
                .chain(self.left_child().subtree().into_iter())
                .chain(self.right_child().subtree().into_iter())
                .collect()
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
    pub fn distance_to_indexed_instance(&self, index: usize) -> f64 {
        self.space().one_to_one(self.arg_center(), index)
    }

    /// Distance from the `center` to the given instance.
    pub fn distance_to_instance(&self, instance: &[T]) -> f64 {
        self.space().metric().one_to_one(self.center(), instance)
    }

    /// Distance from the `center` of this `Cluster` to the center of the
    /// `other` `Cluster`.
    pub fn distance_to_other(&self, other: &Self) -> f64 {
        self.distance_to_indexed_instance(other.arg_center())
    }

    // pub fn add_instance(self: Arc<Self>, index: usize) -> Result<Vec<Arc<Self>>, String> {
    //     if self.depth() == 0 {
    //         Ok(self.__add_instance(index))
    //     } else {
    //         Err("Cannot add an instance to a non-root Cluster.".to_string())
    //     }
    // }

    // fn __add_instance(self: Arc<Self>, index: usize) -> Vec<Arc<Self>> {
    //     let mut result = vec![self.clone()];

    //     if !self.is_leaf() {
    //         let distance_to_left = self.left_child().distance_to_indexed_instance(index);
    //         let distance_to_right = self.right_child().distance_to_indexed_instance(index);

    //         match distance_to_left.partial_cmp(&distance_to_right).unwrap() {
    //             std::cmp::Ordering::Less => result.extend(self.left_child().__add_instance(index).into_iter()),
    //             std::cmp::Ordering::Equal => (),
    //             std::cmp::Ordering::Greater => result.extend(self.right_child().__add_instance(index).into_iter()),
    //         };
    //     }

    //     *self.cardinality.write().unwrap() += 1;

    //     result
    // }
}

#[cfg(test)]
mod tests {
    use crate::dataset::Tabular;
    use crate::prelude::*;
    use crate::traits::space::TabularSpace;

    #[test]
    fn test_cluster() {
        let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
        let dataset = Tabular::<f64>::new(&data, "test_cluster".to_string());
        let metric = metric_from_name::<f64>("euclidean", false).unwrap();
        let space = TabularSpace::new(&dataset, metric.as_ref());
        let partition_criteria = crate::PartitionCriteria::new(true)
            .with_max_depth(3)
            .with_min_cardinality(1);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 4);
        assert_eq!(cluster.num_descendants(), 6);
        assert!(cluster.radius() > 0.);

        assert_eq!(format!("{cluster}"), "1");

        let [left, right] = cluster.children();
        assert_eq!(format!("{left}"), "10");
        assert_eq!(format!("{right}"), "11");

        for child in cluster.children() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 2);
            assert_eq!(child.num_descendants(), 2);
        }
    }
}
