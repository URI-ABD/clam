//! CLAM-Cluster
//!
//! Define and implement the `Cluster` struct.

use std::hash::Hash;
use std::hash::Hasher;

use bitvec::prelude::*;

use crate::prelude::*;
use crate::utils::helpers;

const SUB_SAMPLE_LIMIT: usize = 100;

pub type Ratios = [f64; 6];
type BCluster<'a, T, U> = Box<Cluster<'a, T, U>>;

/// A collection of similar `Instances` from a `Dataset`.
///
/// `Clusters` can be unwieldy to use directly unless you have a
/// good grasp on the underlying invariants.
/// We anticipate that most users' needs will be well met by the higher-level
/// abstractions.
#[derive(Debug, Clone)]
pub struct Cluster<'a, T: Number, U: Number> {
    space: &'a dyn Space<T, U>,
    cardinality: usize,
    indices: Option<Vec<usize>>,
    name: BitVec,
    arg_samples: Option<Vec<usize>>,
    arg_center: Option<usize>,
    arg_radius: Option<usize>,
    radius: Option<U>,
    lfd: Option<f64>,
    left_child: Option<BCluster<'a, T, U>>,
    right_child: Option<BCluster<'a, T, U>>,
    ratios: Option<Ratios>,
}

impl<'a, T: Number, U: Number> PartialEq for Cluster<'a, T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<'a, T: Number, U: Number> Eq for Cluster<'a, T, U> {}

/// Clusters can be sorted based on their name. Sorting a tree of clusters will leave them in the order of a breadth-first traversal.
impl<'a, T: Number, U: Number> PartialOrd for Cluster<'a, T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.depth() == other.depth() {
            self.name.partial_cmp(&other.name)
        } else {
            Some(self.depth().cmp(&other.depth()))
        }
    }
}

impl<'a, T: Number, U: Number> Ord for Cluster<'a, T, U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Clusters are hashed by their names. This means that a hach is only usique within a single tree.
///
/// TODO: Add a tree-based prefix to the cluster names when we need to hash clusters from different trees into the same container.
impl<'a, T: Number, U: Number> Hash for Cluster<'a, T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl<'a, T: Number, U: Number> std::fmt::Display for Cluster<'a, T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name_str: Vec<_> = self.name.iter().map(|b| if *b { "1" } else { "0" }).collect();
        write!(f, "{}", name_str.join(""))
    }
}

impl<'a, T: Number, U: Number> Cluster<'a, T, U> {
    /// Creates a new root `Cluster` on the entire dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset`: A reference to a struct that implements the `Dataset` trait.
    pub fn new_root(space: &'a dyn Space<T, U>) -> Self {
        let name = bitvec![1];
        let indices = space.data().indices();
        Cluster::new(space, indices, name)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `dataset`: A reference to a struct that implements the `Dataset` trait.
    ///   We never copy `Instances` from the `Dataset`.
    /// * `indices`: The `Indices` of `Instances` from the dataset that are contained in the `Cluster`.
    /// * `name`: BitVec name for the `Cluster`.
    /// * `parent`: Weak ref to the parent `Cluster`. Should be `None` for the root.
    pub fn new(space: &'a dyn Space<T, U>, indices: Vec<usize>, name: BitVec) -> Self {
        Cluster {
            space,
            cardinality: indices.len(),
            indices: Some(indices),
            name,
            arg_samples: None,
            arg_center: None,
            arg_radius: None,
            radius: None,
            lfd: None,
            left_child: None,
            right_child: None,
            ratios: None,
        }
    }

    pub fn build(mut self) -> Self {
        let indices = self.indices.clone().unwrap();

        let arg_samples = if indices.len() < SUB_SAMPLE_LIMIT {
            indices.clone()
        } else {
            let n = ((indices.len() as f64).sqrt()) as usize;
            self.space.choose_unique(n, &indices)
        };

        let sample_distances: Vec<U> = self
            .space
            .distance_pairwise(&arg_samples)
            .iter()
            .map(|v| v.iter().cloned().sum())
            .collect();
        let arg_center = arg_samples[helpers::argmin(&sample_distances).0];

        let center_distances = self.space.distance_one_to_many(arg_center, &indices);
        let (arg_radius, radius) = helpers::argmax(&center_distances);
        let arg_radius = indices[arg_radius];

        let lfd = if radius == U::zero() {
            1.
        } else {
            let half_count = center_distances
                .into_iter()
                .filter(|&d| d <= (radius / U::from(2u64).unwrap()))
                .count();
            if half_count > 0 {
                ((indices.len() as f64) / (half_count as f64)).log2()
            } else {
                1.
            }
        };

        self.arg_samples = Some(arg_samples);
        self.arg_center = Some(arg_center);
        self.arg_radius = Some(arg_radius);
        self.radius = Some(radius);
        self.lfd = Some(lfd);

        self
    }

    fn partition_once(&self) -> [Self; 2] {
        let indices = self.indices.clone().unwrap();

        let left_pole = self.arg_radius();
        let remaining_indices: Vec<usize> = indices.iter().filter(|&&i| i != left_pole).cloned().collect();
        let left_distances = self.space.distance_one_to_many(left_pole, &remaining_indices).to_vec();

        let arg_right = helpers::argmax(&left_distances).0;
        let right_pole = remaining_indices[arg_right];

        let (left_indices, right_indices) = if remaining_indices.len() > 1 {
            let left_distances: Vec<U> = left_distances
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != arg_right)
                .map(|(_, &d)| d)
                .collect();
            let remaining_indices: Vec<usize> = remaining_indices
                .iter()
                .filter(|&&i| i != right_pole)
                .cloned()
                .collect();
            let right_distances = self.space.distance_one_to_many(right_pole, &remaining_indices).to_vec();

            let is_closer_to_left_pole: Vec<bool> = left_distances
                .iter()
                .zip(right_distances.iter())
                .map(|(l, r)| l <= r)
                .collect();

            let left_indices = {
                let mut indices: Vec<usize> = remaining_indices
                    .iter()
                    .zip(is_closer_to_left_pole.iter())
                    .filter(|(_, &b)| b)
                    .map(|(&i, _)| i)
                    .collect();
                indices.push(left_pole);
                indices
            };

            let right_indices = {
                let mut indices: Vec<usize> = remaining_indices
                    .iter()
                    .zip(is_closer_to_left_pole.iter())
                    .filter(|(_, &b)| !b)
                    .map(|(&i, _)| i)
                    .collect();
                indices.push(right_pole);
                indices
            };

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

    pub fn partition(mut self, partition_criteria: &criteria::PartitionCriteria<T, U>, recursive: bool) -> Self {
        if partition_criteria.check(&self) {
            let [left, right] = self.partition_once();

            let (left, right) = if recursive {
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

    pub fn with_ratios(mut self, normalized: bool) -> Self {
        if self.depth() > 0 {
            panic!("Cluster Ratios may only be set from the root cluster.")
        }
        if self.is_leaf() {
            panic!("Please build and partition the tree before setting cluster ratios.")
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
            let stds: [f64; 6] = ratios
                .iter()
                .zip(means.iter())
                .map(|(values, &mean)| 1e-8 + helpers::std(values, mean))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            self.set_normalized_ratios(means, stds)
        } else {
            self
        }
    }

    fn set_normalized_ratios(mut self, means: Ratios, stds: Ratios) -> Self {
        let ratios: Vec<_> = self
            .ratios
            .unwrap()
            .into_iter()
            .zip(means.into_iter())
            .zip(stds.into_iter())
            .map(|((value, mean), std)| (value - mean) / (std * 2_f64.sqrt()))
            .map(libm::erf)
            .map(|v| (1. + v) / 2.)
            .collect();
        self.ratios = Some(ratios.try_into().unwrap());

        if !self.is_leaf() {
            self.left_child = Some(Box::new(self.left_child.unwrap().set_normalized_ratios(means, stds)));
            self.right_child = Some(Box::new(self.right_child.unwrap().set_normalized_ratios(means, stds)));
        }
        self
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

    #[inline]
    fn next_ema(&self, ratio: f64, parent_ema: f64) -> f64 {
        let alpha = 2. / 11.;
        alpha * ratio + (1. - alpha) * parent_ema
    }

    pub fn space(&self) -> &dyn Space<T, U> {
        self.space
    }

    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

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

    pub fn name(&self) -> BitVec {
        self.name.clone()
    }

    pub fn is_root(&self) -> bool {
        self.name.len() == 1
    }

    pub fn depth(&self) -> usize {
        self.name.len() - 1
    }

    pub fn arg_samples(&self) -> Vec<usize> {
        self.arg_samples
            .clone()
            .expect("Please call `build` on this cluster before using this method.")
    }

    pub fn arg_center(&self) -> usize {
        self.arg_center
            .expect("Please call `build` on this cluster before using this method.")
    }

    pub fn center(&self) -> Vec<T> {
        self.space.data().get(self.arg_center())
    }

    pub fn arg_radius(&self) -> usize {
        self.arg_radius
            .expect("Please call `build` on this cluster before using this method.")
    }

    pub fn radius(&self) -> U {
        self.radius
            .expect("Please call `build` on this cluster before using this method.")
    }

    pub fn is_singleton(&self) -> bool {
        self.radius() == U::zero()
    }

    pub fn lfd(&self) -> f64 {
        self.lfd
            .expect("Please call `build` on this cluster before using this method.")
    }

    pub fn ratios(&self) -> Ratios {
        self.ratios.expect("Please call `build` before using this method.")
    }

    pub fn children(&self) -> [&Self; 2] {
        [self.left_child(), self.right_child()]
    }

    pub fn left_child(&self) -> &Self {
        self.left_child
            .as_ref()
            .expect("This cluster is a leaf and has no children.")
    }

    pub fn right_child(&self) -> &Self {
        self.right_child
            .as_ref()
            .expect("This cluster is a leaf and has no children.")
    }

    pub fn is_leaf(&self) -> bool {
        self.left_child.is_none()
    }

    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.name.iter().zip(other.name.iter()).all(|(l, r)| *l == *r)
    }

    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

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

    pub fn num_descendants(&self) -> usize {
        self.subtree().len() - 1
    }

    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    pub fn distance_to_indexed_instance(&self, index: usize) -> U {
        self.space().distance_one_to_one(self.arg_center(), index)
    }

    pub fn distance_to_instance(&self, instance: &[T]) -> U {
        self.space().metric().one_to_one(&self.center(), instance)
    }

    pub fn distance_to_other(&self, other: &Self) -> U {
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
        let metric = metric_from_name::<f64, f64>("euclidean").unwrap();
        let space = TabularSpace::new(&dataset, metric, false);
        let partition_criteria = criteria::PartitionCriteria::new(true)
            .with_max_depth(3)
            .with_min_cardinality(1);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 4);
        assert_eq!(cluster.num_descendants(), 6);
        assert!(cluster.radius() > 0.);

        assert_eq!(format!("{:}", cluster), "1");

        // let cluster_str = vec![
        //     "Cluster {".to_string(),
        //     format!("space: {:?},", cluster.space()),
        //     format!("name: {:?},", cluster.name),
        //     format!("cardinality: {:?},", cluster.cardinality),
        //     format!("indices: {:?},", cluster.indices),
        //     format!("argcenter: {:?},", cluster.argcenter),
        //     format!("argradius: {:?},", cluster.argradius),
        //     format!("radius: {:?},", cluster.radius),
        //     format!("lfd: {:?},", cluster.lfd),
        //     format!("children: {:?},", cluster.children),
        //     format!("parent: {:?},", cluster.parent),
        //     format!("ratios: {:?}", cluster.ratios),
        //     "}".to_string(),
        // ]
        // .join(" ");
        // assert_eq!(format!("{:?}", cluster), cluster_str);

        let [left, right] = cluster.children();
        assert_eq!(format!("{:}", left), "10");
        assert_eq!(format!("{:}", right), "11");

        for child in cluster.children() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 2);
            assert_eq!(child.num_descendants(), 2);
        }
    }

    // #[test]
    // fn test_ancestry() {
    //     let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
    //     let metric = metric_from_name("euclidean").unwrap();
    //     let dataset = Arc::new(Tabular::<f64>::new(Arc::new(data), "test_cluster".to_string()));
    //     let space: Arc<dyn Space<f64, f64>> = Arc::new(TabularSpace::new(dataset, metric, false));
    //     let criteria = vec![criteria::max_depth(3), criteria::min_cardinality(1)];
    //     let cluster = Cluster::new_root(Arc::clone(&space))
    //         .build()
    //         .iterative_partition(&criteria);
    //     let (left, right) = cluster.children().unwrap();

    //     let left_ancestry = left.ancestry();
    //     assert_eq!(1, left_ancestry.len());

    //     let right_ancestry = right.ancestry();
    //     assert_eq!(1, right_ancestry.len());
    // }
}
