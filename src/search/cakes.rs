use std::collections::HashMap;
use std::sync::Arc;

use bitvec::prelude::*;
use rayon::prelude::*;

use crate::prelude::*;

/// CLAM-Augmented K-nearest-neighbors Entropy-scaling Search
///
/// Provides tools for similarity search.
/// Search time scales sub-linearly with the size of the dataset.
/// This is orders of magnitude faster than state-of-the-art tools while also
/// guaranteeing exact results (as compared to naive linear search)
/// when the distance function used obeys the triangle inequality.
///
/// Paper pending...
///
/// TODO: Add Compression and Decompression for the dataset and search tree.
///
/// TODO: Add Serde support for storing and loading the search tree.
pub struct Cakes<T: Number, U: Number> {
    /// An Arc to any struct that implements the `Dataset` trait.
    pub dataset: Arc<dyn Dataset<T, U>>,

    /// The root Cluster of the search tree.
    pub root: Arc<Cluster<T, U>>,
}

impl<T: Number, U: Number> std::fmt::Debug for Cakes<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Search").field("dataset", &self.dataset).finish()
    }
}

impl<T: 'static + Number, U: 'static + Number> Cakes<T, U> {
    /// Builds a search tree for the given dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - An Arc to any struct that implements the `Dataset` trait.
    /// * `max_depth` - Clusters in the tree that have a higher depth will not be partitioned.
    ///                 Capped at 63 until I feel like bothering with bit-vectors.
    /// * `min_cardinality` - Clusters in the tree that have a smaller cardinality will not be partitioned.
    pub fn build(dataset: Arc<dyn Dataset<T, U>>, max_depth: Option<usize>, min_cardinality: Option<usize>) -> Cakes<T, U> {
        // parse the max-depth and min-cardinality and create the partition-criterion.
        let criteria = vec![
            criteria::max_depth(max_depth.unwrap_or(50)),
            criteria::min_cardinality(min_cardinality.unwrap_or(1)),
        ];
        // build the search tree.
        let root = Cluster::new(Arc::clone(&dataset), bitvec![Lsb0, u8; 1], dataset.indices()).partition(&criteria);
        // return the struct
        Cakes { dataset, root: Arc::new(root) }
    }

    /// Builds the tree in batches using the memory-fraction provided.
    ///
    /// # Arguments
    ///
    /// * `dataset` - An Arc to any struct that implements the `Dataset` trait.
    /// * `batch_fraction` - The fraction of memory to dedicate to a batch.
    ///                      Defaults to 0.5
    /// * `max_depth` - Clusters in the tree that have a higher depth will not be partitioned.
    ///                 Capped at 63 until I feel like bothering with bit-vectors.
    /// * `min_cardinality` - Clusters in the tree that have a smaller cardinality will not be partitioned.
    pub fn build_in_batches(dataset: Arc<dyn Dataset<T, U>>, batch_fraction: Option<f32>, max_depth: Option<usize>, min_cardinality: Option<usize>) -> Self {
        let batch_size = dataset.batch_size(batch_fraction);
        let (sample_indices, complement_indices) = dataset.subsample_indices(batch_size);

        let subset = dataset.row_major_subset(&sample_indices);
        let mut cakes = Cakes::build(subset, max_depth, min_cardinality);

        let mut flat_tree: Vec<_> = {
            let mut tree = cakes.root.flatten_tree();
            tree.push(Arc::clone(&cakes.root));
            tree.into_par_iter()
                .map(|cluster| {
                    let indices: Vec<_> = cluster.indices.par_iter().map(|&i| sample_indices[i]).collect();
                    let argcenter = sample_indices[cluster.argcenter];
                    let argradius = sample_indices[cluster.argcenter];

                    Arc::new(Cluster {
                        dataset: Arc::clone(&dataset),
                        name: cluster.name.clone(),
                        cardinality: indices.len(),
                        indices,
                        argcenter,
                        argradius,
                        radius: cluster.radius,
                        children: None,
                    })
                })
                .collect()
        };

        let num_batches = {
            let mut num_batches = complement_indices.len() / batch_size;
            if complement_indices.len() % batch_size != 0 {
                num_batches += 1;
            }
            num_batches
        };

        for (i, batch_indices) in complement_indices.chunks(batch_size).enumerate() {
            println!("Inserting batch {} of {} with {} instances", i + 1, num_batches, batch_indices.len());
            let batch_dataset = dataset.row_major_subset(batch_indices);

            // Build a sparse matrix of cluster insertions.
            // | Sequence | Cluster 0                   | Cluster 1  |
            // | seq_00   | None (Not added to cluster) | Some(dist) |
            // | seq_01   | Some(dist)                  | None       |
            let insertion_paths: Vec<Vec<_>> = batch_indices
                .par_iter()
                .enumerate()
                .map(|(i, _)| {
                    let instance = batch_dataset.instance(i);
                    let distance = cakes.root.dataset.metric().distance(&cakes.root.center(), &instance);
                    let insertion_path = cakes.root.add_instance(&instance, distance);

                    flat_tree
                        .par_iter()
                        .map(|cluster| {
                            if insertion_path.contains_key(&cluster.name) {
                                Some(*insertion_path.get(&cluster.name).unwrap())
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect();

            // Reduce the matrix to find the maximum
            let new_radii: Vec<_> = flat_tree
                .par_iter()
                .enumerate()
                .map(|(i, _)| {
                    let temp: Vec<_> = insertion_paths.par_iter().map(|inner| inner[i].unwrap_or_else(U::zero)).collect();
                    crate::utils::argmax(&temp)
                })
                .collect();

            let insertions: Vec<Vec<usize>> = flat_tree
                .par_iter()
                .enumerate()
                .map(|(i, _)| {
                    let temp: Vec<Option<usize>> = insertion_paths
                        .par_iter()
                        .enumerate()
                        .map(|(j, inner)| {
                            let distance = inner[i];
                            if distance.is_some() {
                                Some(j)
                            } else {
                                None
                            }
                        })
                        .collect();
                    temp.into_par_iter().filter(|&v| v.is_some()).map(|v| v.unwrap()).collect()
                })
                .collect();

            let unstacked_tree: Vec<_> = flat_tree
                .par_iter()
                .zip(new_radii.into_par_iter())
                .zip(insertions.into_par_iter())
                .map(|((cluster, (argradius, radius)), indices)| {
                    let mut indices: Vec<_> = indices.into_iter().map(|i| batch_indices[i]).collect();
                    indices.extend(cluster.indices.iter());

                    let (argradius, radius) = if radius > cluster.radius {
                        (batch_indices[argradius], radius)
                    } else {
                        (cluster.argradius, cluster.radius)
                    };

                    Arc::new(Cluster {
                        dataset: Arc::clone(&dataset),
                        name: cluster.name.clone(),
                        cardinality: indices.len(),
                        indices,
                        argcenter: cluster.argcenter,
                        argradius,
                        radius,
                        children: None,
                    })
                })
                .collect();

            cakes = Cakes {
                dataset: Arc::clone(&dataset),
                root: restack_tree(unstacked_tree),
            };

            flat_tree = cakes.root.flatten_tree();
            flat_tree.push(Arc::clone(&cakes.root));
        }

        cakes
    }

    /// Returns the diameter of the search tree, a useful property for judging appropriate search radii.
    pub fn diameter(&self) -> U {
        U::from(2).unwrap() * self.root.radius
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        self.dataset.metric().distance(x, y)
    }

    pub fn knn_indices(&self, query: &[T], k: usize) -> Vec<Index> {
        self.knn(query, k).into_iter().map(|(i, _)| i).collect()
    }

    pub fn knn(&self, _query: &[T], _k: usize) -> Vec<(Index, U)> {
        todo!()
    }

    pub fn rnn_indices(&self, query: &[T], radius: Option<U>) -> Vec<Index> {
        self.rnn(query, radius).into_iter().map(|(i, _)| i).collect()
    }

    /// Performs accelerated rho-nearest search on the dataset and
    /// returns all hits inside a sphere of the given `radius` centered at the requested `query`.
    pub fn rnn(&self, query: &[T], radius: Option<U>) -> Vec<(Index, U)> {
        let (mut definites, potentials) = self.tree_search(query, radius);
        definites.append(&mut self.leaf_search(query, radius, &potentials));
        definites
    }

    /// Performs coarse-grained tree-search to find all instances that are definite or potential hits.
    pub fn tree_search(&self, query: &[T], radius: Option<U>) -> (Vec<(Index, U)>, Vec<Index>) {
        // parse the search radius
        let radius = radius.unwrap_or_else(U::zero);
        // if query ball has overlapping volume with the root, delegate to the recursive, private method.
        let distance = self.distance(&self.root.center(), query);
        if distance <= (radius + self.root.radius) {
            self._tree_search(&self.root, query, radius, distance)
        } else {
            // otherwise, there are no possible hits.
            (Vec::new(), Vec::new())
        }
    }

    fn _tree_search(&self, cluster: &Arc<Cluster<T, U>>, query: &[T], radius: U, distance: U) -> (Vec<(Index, U)>, Vec<Index>) {
        // Invariant: Entering this function means that the current cluster has overlapping volume with the query-ball.
        // Invariant: Triangle-inequality guarantees exactness of results from each recursive call.
        let (mut definites, potentials) = match &cluster.children {
            // There are children. Make recursive calls if necessary.
            Some((left_cluster, right_cluster)) => {
                // get the two vectors of hits from up to two recursive calls.
                let ((mut left_definites, mut left_potentials), (mut right_definites, mut right_potentials)) = rayon::join(
                    || {
                        // If the child has overlap with the query-ball, recurse into the child
                        let distance = self.distance(query, &left_cluster.center());
                        if distance <= (radius + left_cluster.radius) {
                            self._tree_search(left_cluster, query, radius, distance)
                        } else {
                            // otherwise return an empty vec.
                            (Vec::new(), Vec::new())
                        }
                    },
                    || {
                        let distance = self.distance(query, &right_cluster.center());
                        if distance <= (radius + right_cluster.radius) {
                            self._tree_search(right_cluster, query, radius, distance)
                        } else {
                            (Vec::new(), Vec::new())
                        }
                    },
                );
                left_definites.append(&mut right_definites);
                left_potentials.append(&mut right_potentials);
                (left_definites, left_potentials)
            }
            None => {
                // There are no children so return the indices of the current cluster.
                (Vec::new(), cluster.indices.clone())
            }
        };
        // decide whether the cluster-center is a definite hit.
        if distance <= radius {
            definites.push((cluster.argcenter, distance));
        }
        (definites, potentials)
    }

    /// Exhaustively searches the clusters identified by tree-search and
    /// returns a Vec of indices of all hits and their distance from the query.
    pub fn leaf_search(&self, query: &[T], radius: Option<U>, indices: &[Index]) -> Vec<(Index, U)> {
        self.linear_search(query, radius, indices)
    }

    pub fn linear_search_indices(&self, query: &[T], radius: Option<U>, indices: &[Index]) -> Vec<Index> {
        self.linear_search(query, radius, indices).into_iter().map(|(i, _)| i).collect()
    }

    /// Naive search. Useful for leaf-search and for measuring acceleration from entropy-scaling search.
    pub fn linear_search(&self, query: &[T], radius: Option<U>, indices: &[Index]) -> Vec<(Index, U)> {
        let radius = radius.unwrap_or_else(U::zero);
        indices
            .par_iter()
            .map(|&index| (index, self.distance(query, &self.dataset.instance(index))))
            .filter(|(_, d)| *d <= radius)
            .collect()
    }
}

fn child_names<T: Number, U: Number>(cluster: &Arc<Cluster<T, U>>) -> (ClusterName, ClusterName) {
    let mut left_name = cluster.name.clone();
    left_name.push(false);

    let mut right_name = cluster.name.clone();
    right_name.push(true);

    (left_name, right_name)
}

/// Given an unstacked tree as a HashMap of Clusters, rebuild all parent-child relationships and return the root cluster.
pub fn restack_tree<T: Number, U: Number>(tree: Vec<Arc<Cluster<T, U>>>) -> Arc<Cluster<T, U>> {
    let depth = tree.par_iter().map(|c| c.depth()).max().unwrap();
    let mut tree: HashMap<_, _> = tree.into_par_iter().map(|c| (c.name.clone(), c)).collect();

    for d in (0..depth).rev() {
        let (leaves, mut ancestors): (HashMap<_, _>, HashMap<_, _>) = tree.drain().partition(|(_, v)| v.depth() == d + 1);
        let (parents, mut ancestors): (HashMap<_, _>, HashMap<_, _>) = ancestors.drain().partition(|(_, v)| v.depth() == d);

        let parents: HashMap<_, _> = parents
            .par_iter()
            .map(|(_, cluster)| {
                let (left_name, right_name) = child_names(cluster);

                let (children, indices) = if leaves.contains_key(&left_name) {
                    let left = Arc::clone(leaves.get(&left_name).unwrap());
                    let right = Arc::clone(leaves.get(&right_name).unwrap());

                    let mut indices = left.indices.clone();
                    indices.append(&mut right.indices.clone());

                    (Some((left, right)), indices)
                } else {
                    (None, cluster.indices.clone())
                };

                let cluster = Arc::new(Cluster {
                    dataset: Arc::clone(&cluster.dataset),
                    name: cluster.name.clone(),
                    cardinality: indices.len(),
                    indices,
                    children,
                    argcenter: cluster.argcenter,
                    argradius: cluster.argradius,
                    radius: cluster.radius,
                });

                (cluster.name.clone(), cluster)
            })
            .collect();

        parents.into_iter().for_each(|(name, cluster)| {
            ancestors.insert(name, cluster);
        });

        tree = ancestors;
    }

    assert_eq!(1, tree.len());
    Arc::clone(tree.get(&bitvec![Lsb0, u8; 1]).unwrap())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dataset::RowMajor;
    use crate::prelude::*;
    use crate::utils::read_test_data;

    use super::Cakes;

    #[test]
    fn test_search() {
        let data = vec![vec![0., 0.], vec![1., 1.], vec![2., 2.], vec![3., 3.]];
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::new(data, "euclidean", false).unwrap());
        let search = Cakes::build(Arc::clone(&dataset), None, None);

        let query = &[0., 1.];
        let results = search.rnn_indices(query, Some(1.5));
        assert!(results.len() <= 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));

        let query = Arc::new(search.dataset.instance(1));
        let results = search.rnn_indices(&query, None);
        assert_eq!(results.len(), 1);
        assert!(!results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn test_search_large() {
        let (data, _) = read_test_data();
        let dataset: Arc<dyn Dataset<_, f64>> = Arc::new(RowMajor::new(data, "euclidean", true).unwrap());

        let search = Cakes::build(Arc::clone(&dataset), Some(50), None);

        let search_str = ["Search { ".to_string(), format!("dataset: {:?}", dataset), " }".to_string()].join("");
        assert_eq!(format!("{:?}", search), search_str);

        let radius = Some(search.diameter() / 100.);

        for &q in dataset.indices()[0..10].iter() {
            let query = dataset.instance(q);

            let cakes_results = search.rnn_indices(&query, radius);
            let naive_results = search.linear_search_indices(&query, radius, &dataset.indices());

            let no_extra = cakes_results.iter().all(|i| naive_results.contains(i));
            assert!(no_extra, "had some extras {} / {}", naive_results.len(), cakes_results.len());

            let no_misses = naive_results.iter().all(|i| cakes_results.contains(i));
            assert!(no_misses, "had some misses {} / {}", naive_results.len(), cakes_results.len());
        }
    }
}
