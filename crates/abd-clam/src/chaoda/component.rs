//! A `Component` is a single connected subgraph of a `Graph`.

use std::collections::{BTreeMap, BTreeSet};

use distances::Number;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::{Dataset, Instance};

use super::OddBall;

/// A `ClusterKey` is a unique identifier for an `OddBall` in a `Component`.
type ClusterKey = (usize, usize);
/// A `Neighbors` is a mapping from a `ClusterKey` to the distance between the `OddBall`s.
type Neighbors<U> = BTreeMap<ClusterKey, U>;
/// An `AdjacencyList` is a mapping from a `ClusterKey` to its neighbors.
type AdjacencyList<U> = BTreeMap<ClusterKey, Neighbors<U>>;

/// A `Component` is a single connected subgraph of a `Graph`.
///
/// We break the `Graph` into connected `Component`s because this makes several
/// computations significantly easier to think about and implement.
#[derive(Clone)]
pub struct Component<U: Number> {
    /// The clusters and their neighbors in the `Component`.
    adjacency_list: AdjacencyList<U>,
    /// The total number of points in the `OddBall`s in the `Component`.
    population: usize,
    /// Eccentricity of each `OddBall` in the `Component`.
    eccentricities: Option<Vec<usize>>,
    /// Diameter of the `Component`.
    diameter: Option<usize>,
    /// neighborhood sizes of each `OddBall` in the `Component` at each step through a BFT.
    neighborhood_sizes: Option<BTreeMap<ClusterKey, Vec<usize>>>,
    /// The accumulated child-parent cardinality ratio of each `OddBall` in the `Component`.
    accumulated_cp_car_ratios: BTreeMap<ClusterKey, f32>,
    /// The anomaly properties of the `OddBall`s in the `Component`.
    anomaly_properties: BTreeMap<ClusterKey, Vec<f32>>,
}

impl<U: Number> Component<U> {
    /// Create a new `Component` from a collection of `OddBall`s.
    pub fn new<I: Instance, D: Dataset<I, U>, C: OddBall<U>>(clusters: &[&C], data: &D) -> Vec<Self> {
        let adjacency_list: AdjacencyList<_> = clusters
            .par_iter()
            .enumerate()
            .map(|(i, &c1)| {
                let neighbors = clusters
                    .par_iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .filter_map(|(_, &c2)| {
                        let (r1, r2) = (c1.radius(), c2.radius());
                        let d = c1.distance_to_other(data, c2);
                        if d <= r1 + r2 {
                            Some(((c2.offset(), c2.cardinality()), d))
                        } else {
                            None
                        }
                    })
                    .collect::<Neighbors<_>>();
                ((c1.offset(), c1.cardinality()), neighbors)
            })
            .collect();

        let population = clusters.iter().map(|c| c.cardinality()).sum();
        let accumulated_cp_car_ratios = clusters
            .iter()
            .map(|c| ((c.offset(), c.cardinality()), c.accumulated_cp_car_ratio()))
            .collect();
        let anomaly_properties = clusters
            .iter()
            .map(|c| ((c.offset(), c.cardinality()), c.ratios()))
            .collect();

        let c = Self {
            adjacency_list,
            population,
            eccentricities: None,
            diameter: None,
            neighborhood_sizes: None,
            accumulated_cp_car_ratios,
            anomaly_properties,
        };

        let [mut c, mut other] = c.partition();
        let mut components = vec![c];
        while !other.is_empty() {
            [c, other] = other.partition();
            components.push(c);
        }

        components
    }

    /// Partition the `Component` into two `Component`s.
    ///
    /// The first component is a connected subgraph of the original `Component`
    /// and the second component is the rest of the original `Component`.
    ///
    /// This method is used when first constructing the `Graph` to find the
    /// connected subgraphs of the `Graph`.
    ///
    /// This method is meant to be used in a loop to find all connected subgraphs
    /// of a `Graph`. It resets the internal members of the `Component` that are
    /// computed lazily, i.e. the eccentricities, diameter, and neighborhood sizes.
    fn partition(mut self) -> [Self; 2] {
        // Perform a traversal of the adjacency list to find a connected subgraph.
        let mut visited: BTreeSet<ClusterKey> = BTreeSet::new();
        let mut stack: Vec<ClusterKey> = self.adjacency_list.keys().copied().collect();
        while let Some(k) = stack.pop() {
            // Check if the cluster has already been visited.
            if visited.contains(&k) {
                continue;
            }
            // Mark the cluster as visited.
            visited.insert(k);
            // Add the neighbors of the cluster to the stack to be visited.
            for &j in self.adjacency_list[&k].keys() {
                stack.push(j);
            }
        }

        // Partition the clusters into visited and unvisited clusters.
        let (p1, p2): (AdjacencyList<_>, AdjacencyList<_>) =
            self.adjacency_list.into_iter().partition(|(k, _)| visited.contains(k));
        let p1: AdjacencyList<_> = p1
            .into_iter()
            .map(|(k, n)| {
                let n = n.into_iter().filter(|(j, _)| visited.contains(j)).collect();
                (k, n)
            })
            .collect();
        let p2: AdjacencyList<_> = p2
            .into_iter()
            .map(|(k, n)| {
                let n = n.into_iter().filter(|(j, _)| visited.contains(j)).collect();
                (k, n)
            })
            .collect();

        // Build a component from the clusters that were not visited in the traversal.
        let population = p2.iter().map(|((_, c), _)| c).sum();
        let accumulated_cp_car_ratios = p2.keys().map(|&r| (r, self.accumulated_cp_car_ratios[&r])).collect();
        let anomaly_properties = p2.keys().map(|&r| (r, self.anomaly_properties[&r].clone())).collect();
        let other = Self {
            adjacency_list: p2,
            population,
            eccentricities: None,
            diameter: None,
            neighborhood_sizes: None,
            accumulated_cp_car_ratios,
            anomaly_properties,
        };

        // Set the current component to the visited clusters.
        self.adjacency_list = p1;
        self.population = self.adjacency_list.iter().map(|((_, c), _)| c).sum();
        self.eccentricities = None;
        self.diameter = None;
        self.neighborhood_sizes = None;
        self.accumulated_cp_car_ratios = self
            .adjacency_list
            .keys()
            .map(|&r| (r, self.accumulated_cp_car_ratios[&r]))
            .collect();
        self.anomaly_properties = self
            .adjacency_list
            .keys()
            .map(|&r| (r, self.anomaly_properties[&r].clone()))
            .collect();

        [self, other]
    }

    /// Check if the `Component` has any `OddBall`s.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adjacency_list.is_empty()
    }

    /// Iterate over the `OddBall`s in the `Component`.
    pub fn iter_clusters(&self) -> impl Iterator<Item = &ClusterKey> {
        self.adjacency_list.keys()
    }

    /// Iterate over the lists of neighbors of the `OddBall`s in the `Component`.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = &Neighbors<U>> {
        self.adjacency_list.values()
    }

    /// Iterate over the anomaly properties of the `OddBall`s in the `Component`.
    pub fn iter_anomaly_properties(&self) -> impl Iterator<Item = &Vec<f32>> {
        self.anomaly_properties.values()
    }

    /// Get the number of `OddBall`s in the `Component`.
    #[must_use]
    pub fn cardinality(&self) -> usize {
        self.adjacency_list.len()
    }

    /// Get the total number of points in the `Component`.
    #[must_use]
    pub const fn population(&self) -> usize {
        self.population
    }

    /// Get the diameter of the `Component`.
    pub fn diameter(&mut self) -> usize {
        if self.diameter.is_none() {
            if self.eccentricities.is_none() {
                self.compute_eccentricities();
            }
            let ecc = self
                .eccentricities
                .as_ref()
                .unwrap_or_else(|| unreachable!("We just computed the eccentricities"));
            self.diameter = Some(ecc.iter().copied().max().unwrap_or(0));
        }
        self.diameter
            .unwrap_or_else(|| unreachable!("We just computed the diameter"))
    }

    /// Compute the eccentricity of each `OddBall` in the `Component`.
    pub fn compute_eccentricities(&mut self) {
        self.eccentricities = Some(self.neighborhood_sizes().map(Vec::len).collect());
    }

    /// Get the neighborhood sizes of all `OddBall`s in the `Component`.
    pub fn neighborhood_sizes(&mut self) -> impl Iterator<Item = &Vec<usize>> + '_ {
        if self.neighborhood_sizes.is_none() {
            self.neighborhood_sizes = Some(
                self.adjacency_list
                    .par_iter()
                    .map(|(&k, _)| (k, self.compute_neighborhood_sizes(k)))
                    .collect(),
            );
        }
        self.neighborhood_sizes
            .as_ref()
            .unwrap_or_else(|| unreachable!("We just computed the neighborhood sizes"))
            .values()
    }

    /// Get the cumulative number of neighbors encountered after each step through a BFT.
    fn compute_neighborhood_sizes(&self, k: ClusterKey) -> Vec<usize> {
        let mut visited: BTreeSet<ClusterKey> = BTreeSet::new();
        let mut neighborhood_sizes: Vec<usize> = Vec::new();
        let mut stack: Vec<ClusterKey> = vec![k];

        while let Some(i) = stack.pop() {
            if visited.contains(&i) {
                continue;
            }
            visited.insert(i);
            let new_neighbors = self.adjacency_list[&i]
                .iter()
                .filter(|(j, _)| !visited.contains(j))
                .collect::<Vec<_>>();
            neighborhood_sizes.push(new_neighbors.len());
            stack.extend(new_neighbors.iter().map(|(j, _)| *j));
        }

        neighborhood_sizes
            .iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect()
    }

    /// Compute the stationary probability of each `OddBall` in the `Component`.
    #[must_use]
    pub fn compute_stationary_probabilities(&self, num_steps: usize) -> Vec<f32> {
        if self.cardinality() == 1 {
            // Singleton components need to be marked as anomalous.
            return vec![0.0];
        }

        let mut transition_matrix = vec![0_f32; self.cardinality() * self.cardinality()];
        for (i, (_, neighbors)) in self.adjacency_list.iter().enumerate() {
            for (j, (_, &d)) in neighbors.iter().enumerate() {
                transition_matrix[i * self.cardinality() + j] = d.as_f32().recip();
            }
        }
        // Convert the transition matrix to an Array2
        let mut transition_matrix = Array2::from_shape_vec((self.cardinality(), self.cardinality()), transition_matrix)
            .unwrap_or_else(|e| unreachable!("We created a square Transition matrix: {e}"));

        // Normalize the transition matrix so that each row sums to 1
        for i in 0..self.cardinality() {
            let row_sum = transition_matrix.row(i).sum();
            transition_matrix.row_mut(i).mapv_inplace(|x| x / row_sum);
        }

        // Compute the stationary probabilities by squaring the transition matrix `num_steps` times
        for _ in 0..num_steps {
            transition_matrix = transition_matrix.dot(&transition_matrix);
        }

        // Compute the stationary probabilities by summing the rows of the transition matrix
        transition_matrix.sum_axis(Axis(1)).to_vec()
    }

    /// Get the accumulated child-parent cardinality ratio of each `OddBall` in the `Component`.
    pub fn accumulated_cp_car_ratios(&self) -> impl Iterator<Item = f32> + '_ {
        self.accumulated_cp_car_ratios.values().copied()
    }
}
