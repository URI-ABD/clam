//! A wrapper around the `FlatVec` struct that will be used for the mass-spring
//! system.

use std::collections::{HashMap, HashSet};

use distances::{number::Multiplication, Number};
use rayon::prelude::*;

use crate::{
    cluster::ParCluster,
    dataset::{AssociatesMetadata, AssociatesMetadataMut, ParDataset},
    metric::ParMetric,
    Cluster, Dataset, FlatVec, Metric,
};

use super::{Spring, Vector};

/// A mass-spring system in DIM dimensions, used to perform dimensionality
/// reduction.
///
/// Each item in the original dataset is treated as a unit mass, and the
/// pairwise distances between items are used to determine the spring forces
/// between masses. The system is then simulated to find a lower-dimensional
/// embedding of the dataset.
pub struct System<'a, const DIM: usize, Me, T: Number, C: Cluster<T>> {
    /// Each item in the system is a pair of vectors representing the position
    /// and velocity of the point representing the item in the original dataset.
    data: FlatVec<[Vector<DIM>; 2], Me>,
    /// The `Spring`s that connect clusters which have child clusters.
    springs: Vec<Spring<'a, T, C>>,
    /// The `Spring`s that connect leaf clusters.
    leaf_springs: Vec<Spring<'a, T, C>>,
    /// The damping coefficient of the system.
    beta: f32,
    /// The kinetic energy and potential energy of the system throughout the
    /// simulation.
    energy_history: Vec<(f32, f32)>,
}

impl<const DIM: usize, Me: Clone, T: Number, C: Cluster<T>> System<'_, DIM, Me, T, C> {
    /// Create a new `MassSpringSystem` with the given `Cluster` tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The original dataset.
    /// - `beta`: The damping coefficient of the system. If `None`, the default
    ///   value is `0.999`.
    ///
    /// # Errors
    ///
    /// - If the damping coefficient is not in the range `(0, 1)`.
    pub fn new<I>(data: &FlatVec<I, Me>, beta: Option<f32>) -> Result<Self, String> {
        let items = vec![[Vector::zero(); 2]; data.cardinality()];
        let data = FlatVec::new(items)?.with_metadata(data.metadata())?;
        let springs = Vec::new();
        let leaf_springs = Vec::new();
        let energy_history = vec![(0.0, 0.0)];

        if let Some(beta) = beta {
            Self {
                data,
                springs,
                leaf_springs,
                beta,
                energy_history,
            }
            .with_beta(beta)
        } else {
            let beta = 0.999;
            Ok(Self {
                data,
                springs,
                leaf_springs,
                beta,
                energy_history,
            })
        }
    }

    /// Extract the reduced dataset from the mass-spring system.
    ///
    /// This is a `FlatVec` of `DIM`-dimensional points.
    pub fn extract_positions(&self) -> FlatVec<[f32; DIM], Me> {
        self.data.clone().transform_items(|[p, _]| *p)
    }
}

impl<const DIM: usize, Me: Clone + Send + Sync, T: Number, C: Cluster<T>> System<'_, DIM, Me, T, C> {
    /// Parallel version of [`extract_positions`](Self::extract_positions).
    pub fn par_extract_positions(&self) -> FlatVec<[f32; DIM], Me> {
        self.data.clone().par_transform_items(|[p, _]| *p)
    }
}

impl<'a, const DIM: usize, Me, T: Number, C: Cluster<T>> System<'a, DIM, Me, T, C> {
    /// Change the damping coefficient of the system.
    ///
    /// # Errors
    ///
    /// - If the damping coefficient is not in the range `(0, 1)`.
    pub fn with_beta(mut self, beta: f32) -> Result<Self, String> {
        if beta <= 0.0 || beta >= 1.0 {
            let msg = format!("Damping coefficient must be in the range (0, 1). Got {beta:.2e} instead.");
            ftlog::error!("{msg}");
            Err(msg)
        } else {
            self.beta = beta;
            Ok(self)
        }
    }

    /// Initialize the system with the given root `Cluster`.
    ///
    /// This will create a `Spring` between each pair of child clusters in the
    /// root `Cluster` and assign a different random position and velocity to
    /// the points represented by each child cluster.
    ///
    /// # Arguments
    ///
    /// - `k`: The spring constant of the new springs.
    /// - `root`: The root `Cluster` of the tree.
    /// - `data`: The dataset containing the original items.
    /// - `metric`: The metric to use to calculate the distances between the
    ///   original items.
    /// - `rng`: The random number generator to use in the simulation.
    ///
    /// # Errors
    ///
    /// - If the root `Cluster` has fewer than two children.
    pub fn initialize_with_root<I, D: Dataset<I>, M: Metric<I, T>, R: rand::Rng>(
        &mut self,
        k: f32,
        root: &'a C,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) -> Result<(), String> {
        let children = root.children();
        if children.len() < 2 {
            let msg = "Root cluster must have at least two children.";
            ftlog::error!("{msg}");
            return Err(msg.to_string());
        }

        // For each child, choose a random position inside a hypercube centered
        // at the origin with side length equal to the diameter of the root.
        let radius = root.radius().as_f32();
        for &c in &children {
            let p = Vector::random(rng, -radius, radius);
            let v = Vector::random(rng, -radius, radius);
            for i in c.indices() {
                self[i] = [p, v];
            }
        }

        // For each pair of children, create a spring between them.
        (self.leaf_springs, self.springs) = children
            .iter()
            .flat_map(|&a| {
                children
                    .iter()
                    .filter_map(|&b| {
                        let (i, j) = (a.arg_center(), b.arg_center());
                        if i < j {
                            let l0 = data.one_to_one(i, j, metric);
                            let l = self.distance_between(i, j);
                            Some(Spring::new([a, b], k, l0, l))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .partition(|s| {
                let [a, b] = s.clusters();
                a.is_leaf() && b.is_leaf()
            });

        self.reset_energy_history();

        Ok(())
    }

    /// Simulate the system until the points reach the leaves of the tree.
    ///
    /// # Arguments
    ///
    /// - `k`: The initial spring constant of the springs in the system.
    /// - `data`: See [`initialize_with_root`](Self::initialize_with_root).
    /// - `metric`: See [`initialize_with_root`](Self::initialize_with_root).
    /// - `rng`: The random number generator to use in the simulation.
    /// - `dt`: See [`update_step`](Self::update_step).
    /// - `patience`: See [`simulate_to_stability`](Self::simulate_to_stability).
    /// - `target`: See [`simulate_to_stability`](Self::simulate_to_stability).
    /// - `max_steps`: See [`simulate_to_stability`](Self::simulate_to_stability).
    /// - `dk`: The factor by which to multiplicatively decrease the spring
    ///   constant when replacing a cluster with its children. If `None`, the
    ///   default value is `0.5`. This value must be in the range `(0, 1)`.
    /// - `retention_depth`: The number of times after a cluster is replaced by
    ///   its children that the corresponding springs are retained. If `None`,
    ///   the default value is `3`. Increasing this value will exponentially
    ///   increase the number of springs in the system.
    /// - `f`: The fraction of the most displaced springs to replace by child
    ///   clusters. at each step. If `None`, the default value is `0.1`. This
    ///   value must be in the range `(0, 1)`.
    ///
    /// # Errors
    ///
    /// - If `dk` is not in the range `(0, 1)`.
    /// - If `f` is not in the range `(0, 1)`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        clippy::similar_names,
        clippy::many_single_char_names
    )]
    pub fn simulate_to_leaves<I, D: Dataset<I>, M: Metric<I, T>, R: rand::Rng>(
        &mut self,
        k: f32,
        data: &D,
        metric: &M,
        rng: &mut R,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
        dk: Option<f32>,
        retention_depth: Option<usize>,
        f: Option<f32>,
    ) -> Result<(), String> {
        let dk = if let Some(dk) = dk {
            if dk > 0.0 && dk < 1.0 {
                dk
            } else {
                let msg = format!("`dk` must be in the range (0, 1). Got {dk:.2e} instead.");
                ftlog::error!("{msg}");
                return Err(msg);
            }
        } else {
            0.5
        };

        let min_k = k * dk.powi(retention_depth.unwrap_or(3).as_i32());

        let f = if let Some(f) = f {
            if f > 0.0 && f < 1.0 {
                f
            } else {
                let msg = format!("`f` must be in the range (0, 1). Got {f:.2e} instead.");
                ftlog::error!("{msg}");
                return Err(msg);
            }
        } else {
            0.1
        };

        while !self.springs.is_empty() {
            // Remove springs that are too weak and sort the remaining springs
            // by their displacement ratio.
            self.remove_weak_springs(min_k);
            self.sort_springs_by_displacement();

            // Get the clusters connected to the most displaced springs.
            let at = ((1.0 - f) * self.springs.len().as_f32()).floor().as_usize();
            let stressed_springs = self.springs.split_off(at);
            let stressed_clusters = stressed_springs
                .iter()
                .flat_map(Spring::clusters)
                .collect::<HashSet<_>>();

            // Return the stressed springs to the system.
            self.springs.extend(stressed_springs.iter().copied());

            // Remove all springs connected to stressed clusters.
            let (stressed_springs, disconnected) = self.springs.drain(..).partition(|s| {
                let [a, b] = s.clusters();
                stressed_clusters.contains(&a) || stressed_clusters.contains(&b)
            });
            self.springs = disconnected;

            // Collect all stressed clusters and their connecting springs.
            let mut parents = HashMap::new();
            for s in stressed_springs {
                let ([a, b], k, _, _) = s.deconstruct();
                let k = k * dk;
                if !a.is_leaf() {
                    parents.entry(a).or_insert_with(Vec::new).push((b, k));
                }
                if !b.is_leaf() {
                    parents.entry(b).or_insert_with(Vec::new).push((a, k));
                }
            }

            // Create a triangle for each parent cluster to be replaced.
            let triangles = parents
                .iter()
                .map(|(&c, _)| {
                    let [a, b] = {
                        let children = c.children();
                        [children[0], children[1]]
                    };

                    // Compute the true distances between the child clusters and
                    // the parent cluster.
                    let (ac, bc, ab) = (
                        data.one_to_one(a.arg_center(), c.arg_center(), metric),
                        data.one_to_one(b.arg_center(), c.arg_center(), metric),
                        data.one_to_one(a.arg_center(), b.arg_center(), metric),
                    );

                    // Compute the displacements of the child clusters from the
                    // parent cluster.
                    let (dxa, dxb, dyb) = triangle_displacements(ac, bc, ab);

                    (c, (a, b, ac, bc, ab, dxa, dxb, dyb))
                })
                .collect::<HashMap<_, _>>();

            let triangles = triangles
                .into_iter()
                .map(|(c, (a, b, ac, bc, ab, dxa, dxb, dyb))| {
                    // Compute new positions for the child clusters.
                    let x = Vector::random_unit(rng);
                    let pa = self[a.arg_center()][0] + x * dxa;

                    let y = x.perpendicular(rng);
                    let pb = self[b.arg_center()][0] + x * dxb + y * dyb;

                    // Set the new positions of points in the child clusters.
                    for i in a.indices() {
                        self[i][0] = pa;
                    }
                    for i in b.indices() {
                        self[i][0] = pb;
                    }

                    let ac = Spring::new([a, c], k, ac, self.distance_between(a.arg_center(), c.arg_center()));
                    let bc = Spring::new([b, c], k, bc, self.distance_between(b.arg_center(), c.arg_center()));
                    let ab = Spring::new([a, b], k, ab, self.distance_between(a.arg_center(), b.arg_center()));

                    self.springs.push(ac);
                    self.springs.push(bc);

                    if a.is_leaf() && b.is_leaf() {
                        self.leaf_springs.push(ab);
                    } else {
                        self.springs.push(ab);
                    }

                    (c, [a, b])
                })
                .collect::<HashMap<_, _>>();

            // Create springs between the pairs of child clusters of different
            // parent clusters.
            let (new_springs, new_leaf_springs): (Vec<_>, Vec<_>) = triangles
                .iter()
                .flat_map(|(&c, &[ca, cb])| {
                    parents[&c]
                        .iter()
                        .filter(|&&(d, _)| !d.is_leaf() && c.arg_center() < d.arg_center())
                        .flat_map(|&(d, k)| {
                            let [da, db] = triangles[&d];
                            [
                                self.new_spring([ca, da], k, data.one_to_one(ca.arg_center(), da.arg_center(), metric)),
                                self.new_spring([ca, db], k, data.one_to_one(ca.arg_center(), db.arg_center(), metric)),
                                self.new_spring([cb, da], k, data.one_to_one(cb.arg_center(), da.arg_center(), metric)),
                                self.new_spring([cb, db], k, data.one_to_one(cb.arg_center(), db.arg_center(), metric)),
                            ]
                        })
                })
                .partition(|s| {
                    let [a, b] = s.clusters();
                    a.is_leaf() && b.is_leaf()
                });

            // Add the new springs to the system.
            self.springs.extend(new_springs);
            self.leaf_springs.extend(new_leaf_springs);

            // Simulate the system until it reaches stability.
            self.simulate_to_stability(dt, patience, target, max_steps);
        }

        // Simulate the system for longer to ensure that it has reached a stable
        // state with leaf clusters.
        let patience = patience * 10;
        let target = target.map(|t| t * 0.1);
        let max_steps = max_steps.map(|m| m * 10);
        self.simulate_to_stability(dt, patience, target, max_steps);

        ftlog::info!(
            "Reached instability of {:.2e} with {} leaf springs.",
            self.instability(patience),
            self.leaf_springs.len()
        );

        Ok(())
    }

    /// Adds a new spring between the given clusters.
    pub fn add_spring(&mut self, [a, b]: [&'a C; 2], k: f32, l0: T) {
        let spring = self.new_spring([a, b], k, l0);
        if a.is_leaf() && b.is_leaf() {
            self.leaf_springs.push(spring);
        } else {
            self.springs.push(spring);
        }
    }

    /// Creates a new spring between the given clusters.
    pub fn new_spring(&self, [a, b]: [&'a C; 2], k: f32, l0: T) -> Spring<'a, T, C> {
        Spring::new([a, b], k, l0, self.distance_between(a.arg_center(), b.arg_center()))
    }

    /// Remove springs and leaf-springs whose spring constant is below the given
    /// threshold.
    fn remove_weak_springs(&mut self, min_k: f32) {
        self.springs.retain(|s| s.k() >= min_k);
        self.leaf_springs.retain(|s| s.k() >= min_k);
    }

    /// Sort the springs by their displacement ratio in ascending order.
    fn sort_springs_by_displacement(&mut self) {
        let mut ratios_springs = self.springs.drain(..).map(|s| (s.ratio(), s)).collect::<Vec<_>>();
        ratios_springs.sort_by(|(a, _), (b, _)| a.total_cmp(b));
        self.springs = ratios_springs.into_iter().map(|(_, s)| s).collect();
    }

    /// Get the total kinetic energy of the system.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.data
            .items()
            .iter()
            .map(|[_, v]| v.magnitude().square())
            .sum::<f32>()
            .half()
    }

    /// Get the total potential energy of the system.
    #[must_use]
    pub fn potential_energy(&self) -> f32 {
        self.springs
            .iter()
            .chain(self.leaf_springs.iter())
            .map(Spring::potential_energy)
            .sum()
    }

    /// Update the energies of the system, storing the current kinetic and
    /// potential energies as the last element of the `energies` vector.
    fn update_energy_history(&mut self) {
        let ke = self.kinetic_energy();
        let pe = self.potential_energy();
        self.energy_history.push((ke, pe));
    }

    /// Wipes the energy history of the system, leaving only the current energy
    /// state.
    fn reset_energy_history(&mut self) {
        self.energy_history.clear();
        self.update_energy_history();
    }

    /// Get the instability of the `System` over the last `n` time-steps.
    ///
    /// The instability is the mean of the coefficient of variation of the
    /// kinetic and potential energies over the last `n` time-steps.
    ///
    /// # Arguments
    ///
    /// - `n`: The number of time-steps to consider.
    ///
    /// # Returns
    ///
    /// The instability of the `System` in a [0, inf) range, lower values
    /// indicating higher stability.
    #[must_use]
    pub fn instability(&'a self, n: usize) -> f32 {
        if self.energy_history.len() < n {
            1.0
        } else {
            let (ke_values, pe_values): (Vec<_>, Vec<_>) = self
                .energy_history
                .iter()
                .skip(self.energy_history.len() - n)
                .copied()
                .unzip();

            let var_kinetic: f32 = crate::utils::coefficient_of_variation(&ke_values);
            let var_potential: f32 = crate::utils::coefficient_of_variation(&pe_values);

            (var_kinetic + var_potential).half()
        }
    }

    /// Get the current euclidean distance between two points in the system.
    #[must_use]
    pub fn distance_between(&self, i: usize, j: usize) -> f32 {
        self[i][0].distance_to(&self[j][0])
    }

    /// Updates the `Spring`s in the system to reflect the current positions of
    /// the masses.
    fn update_springs(&mut self) {
        let new_lengths = self
            .springs
            .iter()
            .map(|s| {
                let [a, b] = s.clusters();
                self.distance_between(a.arg_center(), b.arg_center())
            })
            .collect::<Vec<_>>();
        self.springs
            .iter_mut()
            .zip(new_lengths)
            .for_each(|(s, l)| s.update_length(l));
    }

    /// Simulates a single step of the mass-spring system.
    ///
    /// We first calculate the force exerted by each spring, then accumulate the
    /// forces acting on each mass. We then apply damping to the accumulated
    /// forces, and update the positions and velocities of the masses.
    ///
    /// Then, we update the springs in the system to reflect the new positions
    /// of the masses.
    ///
    /// Finally, we update the energy history of the system.
    pub fn update_step(&mut self, dt: f32) {
        // Calculate the force exerted by each spring.
        let forces = self
            .springs
            .iter()
            .chain(self.leaf_springs.iter())
            .flat_map(|s| {
                let [a, b] = s.clusters();
                let f = s.unit_vector(self) * s.f_mag();
                [(a, f), (b, -f)]
            })
            .collect::<Vec<_>>();

        // Calculate the change in velocity for each point.
        let dvs = {
            let mut dvs = vec![Vector::zero(); self.data.cardinality()];

            for (c, f) in forces {
                let m = c.cardinality().as_f32();
                let a = f / m;
                let dv = a * dt;
                for i in c.indices() {
                    // The addition here accounts for the accumulated forces.
                    dvs[i] += dv;
                }
            }

            dvs
        };

        // Apply the changes to the positions and velocities of the points.
        self.data.transform_items_enumerated_in_place(|i, [mut x, mut v]| {
            v += dvs[i];
            x += v * dt;
        });

        // Update the springs in the system.
        self.update_springs();

        // Update the energy history of the system.
        self.update_energy_history();
    }

    /// Simulate the system until it reaches a stable state.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step to use in the simulation.
    /// - `patience`: The number of time-steps to wait for the system to reach
    ///   stability.
    /// - `target`: The stability value to reach before stopping the simulation.
    ///   If `None`, the default value is `1e-5`.
    /// - `max_steps`: The maximum number of time-steps to simulate. If `None`,
    ///   the default value is `usize::MAX`.
    pub fn simulate_to_stability(&mut self, dt: f32, patience: usize, target: Option<f32>, max_steps: Option<usize>) {
        let target = target.unwrap_or(1e-5);
        let max_steps = max_steps.unwrap_or(usize::MAX);

        let mut i = 0;
        let mut instability = self.instability(patience);
        while i < patience || (instability > target && i < max_steps) {
            i += 1;
            self.update_step(dt);
            instability = self.instability(patience);
        }

        ftlog::info!(
            "Reached instability of {instability:.2e} after {i} steps with {} springs.",
            self.springs.len()
        );
    }
}

impl<'a, const DIM: usize, Me: Send + Sync, T: Number, C: ParCluster<T>> System<'a, DIM, Me, T, C> {
    /// Parallel version of [`kinetic_energy`](Self::kinetic_energy).
    #[must_use]
    pub fn par_kinetic_energy(&self) -> f32 {
        self.data
            .items()
            .par_iter()
            .map(|[_, v]| v.magnitude().square())
            .sum::<f32>()
            .half()
    }

    /// Parallel version of [`potential_energy`](Self::potential_energy).
    #[must_use]
    pub fn par_potential_energy(&self) -> f32 {
        self.springs
            .par_iter()
            .chain(self.leaf_springs.par_iter())
            .map(Spring::potential_energy)
            .sum()
    }

    /// Parallel version of [`update_energy_history`](Self::update_energy_history).
    fn par_update_energy_history(&mut self) {
        let ke = self.par_kinetic_energy();
        let pe = self.par_potential_energy();
        self.energy_history.push((ke, pe));
    }

    /// Parallel version of [`reset_energy_history`](Self::reset_energy_history).
    fn par_reset_energy_history(&mut self) {
        self.energy_history.clear();
        self.par_update_energy_history();
    }

    /// Parallel version of [`initialize_with_root`](Self::initialize_with_root).
    ///
    /// # Errors
    ///
    /// - See [`initialize_with_root`](Self::initialize_with_root).
    pub fn par_initialize_with_root<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>, R: rand::Rng>(
        &mut self,
        k: f32,
        root: &'a C,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) -> Result<(), String> {
        let children = root.children();
        if children.len() < 2 {
            let msg = "Root cluster must have at least two children.";
            ftlog::error!("{msg}");
            return Err(msg.to_string());
        }

        // For each child, choose a random position inside a hypercube centered
        // at the origin with side length equal to the diameter of the root.
        let radius = root.radius().as_f32();
        for &c in &children {
            let p = Vector::random(rng, -radius, radius);
            let v = Vector::random(rng, -radius, radius);
            for i in c.indices() {
                self[i] = [p, v];
            }
        }

        // For each pair of children, create a spring between them.
        (self.leaf_springs, self.springs) = children
            .par_iter()
            .flat_map(|&a| {
                children
                    .par_iter()
                    .filter_map(|&b| {
                        let (i, j) = (a.arg_center(), b.arg_center());
                        if i < j {
                            let l0 = data.one_to_one(i, j, metric);
                            let l = self.distance_between(i, j);
                            Some(Spring::new([a, b], k, l0, l))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .partition(|s| {
                let [a, b] = s.clusters();
                a.is_leaf() && b.is_leaf()
            });

        self.par_reset_energy_history();

        Ok(())
    }

    /// Parallel version of [`update_springs`](Self::update_springs).
    fn par_update_springs(&mut self) {
        let new_lengths = self
            .springs
            .par_iter()
            .map(|s| {
                let [a, b] = s.clusters();
                self.distance_between(a.arg_center(), b.arg_center())
            })
            .collect::<Vec<_>>();
        self.springs
            .par_iter_mut()
            .zip(new_lengths)
            .for_each(|(s, l)| s.update_length(l));
    }

    /// Parallel version of [`update_step`](Self::update_step).
    pub fn par_update_step(&mut self, dt: f32) {
        // Calculate the force exerted by each spring.
        let forces = self
            .springs
            .par_iter()
            .chain(self.leaf_springs.par_iter())
            .flat_map(|s| {
                let [a, b] = s.clusters();
                let f = s.unit_vector(self) * s.f_mag();
                [(a, f), (b, -f)]
            })
            .collect::<Vec<_>>();

        // Calculate the change in velocity for each point.
        let dvs = {
            let mut dvs = vec![Vector::zero(); self.data.cardinality()];

            for (c, f) in forces {
                let m = c.cardinality().as_f32();
                let a = f / m;
                let dv = a * dt;
                for i in c.indices() {
                    // The addition here accounts for the accumulated forces.
                    dvs[i] += dv;
                }
            }

            dvs
        };

        // Apply the changes to the positions and velocities of the points.
        self.data.par_transform_items_enumerated_in_place(|i, [mut x, mut v]| {
            v += dvs[i];
            x += v * dt;
        });

        // Update the springs in the system.
        self.par_update_springs();

        // Update the energy history of the system.
        self.par_update_energy_history();
    }

    /// Parallel version of [`simulate_to_stability`](Self::simulate_to_stability).
    pub fn par_simulate_to_stability(
        &mut self,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
    ) {
        let target = target.unwrap_or(1e-5);
        let max_steps = max_steps.unwrap_or(usize::MAX);

        let mut i = 0;
        let mut instability = self.instability(patience);
        while i < patience || (instability > target && i < max_steps) {
            i += 1;
            self.update_step(dt);
            instability = self.instability(patience);
        }

        ftlog::info!(
            "Reached instability of {instability:.2e} after {i} steps with {} springs and {} leaf springs.",
            self.springs.len(),
            self.leaf_springs.len()
        );
    }

    /// Parallel version of [`simulate_to_leaves`](Self::simulate_to_leaves).
    ///
    /// # Errors
    ///
    /// - See [`simulate_to_leaves`](Self::simulate_to_leaves).
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        clippy::similar_names,
        clippy::many_single_char_names
    )]
    pub fn par_simulate_to_leaves<
        I: Send + Sync,
        D: ParDataset<I>,
        M: ParMetric<I, T>,
        R: rand::Rng,
        P: AsRef<std::path::Path>,
    >(
        &mut self,
        k: f32,
        data: &D,
        metric: &M,
        rng: &mut R,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
        dk: Option<f32>,
        retention_depth: Option<usize>,
        f: Option<f32>,
        out_dir: &P,
    ) -> Result<(), String>
    where
        Me: Clone,
    {
        let dk = if let Some(dk) = dk {
            if dk > 0.0 && dk < 1.0 {
                dk
            } else {
                let msg = format!("`dk` must be in the range (0, 1). Got {dk:.2e} instead.");
                ftlog::error!("{msg}");
                return Err(msg);
            }
        } else {
            0.5
        };

        let min_k = k * dk.powi(retention_depth.unwrap_or(3).as_i32());

        let f = if let Some(f) = f {
            if f > 0.0 && f < 1.0 {
                f
            } else {
                let msg = format!("`f` must be in the range (0, 1). Got {f:.2e} instead.");
                ftlog::error!("{msg}");
                return Err(msg);
            }
        } else {
            0.1
        };

        let mut i = 0;
        while !self.springs.is_empty() {
            i += 1;

            // Remove springs that are too weak and sort the remaining springs
            // by their displacement ratio.
            self.remove_weak_springs(min_k);
            self.sort_springs_by_displacement();

            // Get the clusters connected to the most displaced springs.
            let at = ((1.0 - f) * self.springs.len().as_f32()).floor().as_usize();
            let stressed_springs = self.springs.split_off(at);
            ftlog::info!(
                "Step {i}, Removed {} springs leaving {}.",
                stressed_springs.len(),
                self.springs.len()
            );

            let stressed_parents = stressed_springs
                .par_iter()
                .flat_map(Spring::clusters)
                .filter(|&c| !c.is_leaf())
                .collect::<HashSet<_>>();
            ftlog::info!("Step {i}, Found {} stressed parent clusters.", stressed_parents.len());

            // Return the stressed springs to the system.
            self.springs.extend(stressed_springs.iter().copied());

            // Remove all springs connected to stressed parents.
            let (connected, disconnected): (Vec<_>, _) = self.springs.par_drain(..).partition(|s| {
                let [a, b] = s.clusters();
                stressed_parents.contains(&a) || stressed_parents.contains(&b)
            });
            self.springs = disconnected;
            ftlog::info!(
                "Step {i}, Removed {} stressed springs leaving {}",
                connected.len(),
                self.springs.len()
            );

            // Collect all stressed clusters and their connecting springs.
            let mut parents = HashMap::new();
            for s in connected {
                let ([a, b], k, _, _) = s.deconstruct();
                let k = k * dk;
                if !a.is_leaf() {
                    parents.entry(a).or_insert_with(Vec::new).push((b, k));
                }
                if !b.is_leaf() {
                    parents.entry(b).or_insert_with(Vec::new).push((a, k));
                }
            }
            ftlog::info!("Step {i}, Found {} stressed clusters.", parents.len());

            // Create a triangle for each parent cluster to be replaced.
            let triangles = parents
                .par_iter()
                .map(|(&c, _)| {
                    let [a, b] = {
                        let children = c.children();
                        [children[0], children[1]]
                    };

                    // Compute the true distances between the child clusters and
                    // the parent cluster.
                    let (ac, bc, ab) = (
                        data.par_one_to_one(a.arg_center(), c.arg_center(), metric),
                        data.par_one_to_one(b.arg_center(), c.arg_center(), metric),
                        data.par_one_to_one(a.arg_center(), b.arg_center(), metric),
                    );

                    // Compute the displacements of the child clusters from the
                    // parent cluster.
                    let (dxa, dxb, dyb) = triangle_displacements(ac, bc, ab);

                    (c, (a, b, ac, bc, ab, dxa, dxb, dyb))
                })
                .collect::<HashMap<_, _>>();

            let triangles = triangles
                .into_iter()
                .map(|(c, (a, b, ac, bc, ab, dxa, dxb, dyb))| {
                    // Compute new positions for the child clusters.
                    let x = Vector::random_unit(rng);
                    let pa = self[a.arg_center()][0] + x * dxa;

                    let y = x.perpendicular(rng);
                    let pb = self[b.arg_center()][0] + x * dxb + y * dyb;

                    // Set the new positions of points in the child clusters.
                    for i in a.indices() {
                        self[i][0] = pa;
                    }
                    for i in b.indices() {
                        self[i][0] = pb;
                    }

                    self.add_spring([a, c], k, ac);
                    self.add_spring([b, c], k, bc);

                    let ab = self.new_spring([a, b], k, ab);
                    if a.is_leaf() && b.is_leaf() {
                        self.leaf_springs.push(ab);
                    } else {
                        self.springs.push(ab);
                    }

                    (c, [a, b])
                })
                .collect::<HashMap<_, _>>();

            // Create springs between the pairs of child clusters of different
            // parent clusters.
            let (new_leaf_springs, new_springs): (Vec<_>, Vec<_>) = triangles
                .par_iter()
                .flat_map(|(&c, &[ca, cb])| {
                    parents[&c]
                        .par_iter()
                        .filter(|&&(d, _)| !d.is_leaf() && c.arg_center() < d.arg_center())
                        .flat_map(|&(d, k)| {
                            let [da, db] = triangles[&d];
                            [
                                self.new_spring([ca, da], k, data.one_to_one(ca.arg_center(), da.arg_center(), metric)),
                                self.new_spring([ca, db], k, data.one_to_one(ca.arg_center(), db.arg_center(), metric)),
                                self.new_spring([cb, da], k, data.one_to_one(cb.arg_center(), da.arg_center(), metric)),
                                self.new_spring([cb, db], k, data.one_to_one(cb.arg_center(), db.arg_center(), metric)),
                            ]
                        })
                })
                .partition(|s| {
                    let [a, b] = s.clusters();
                    a.is_leaf() && b.is_leaf()
                });
            ftlog::info!(
                "Step {i}, Created {} new leaf springs and {} new springs.",
                new_leaf_springs.len(),
                new_springs.len()
            );

            // Add the new springs to the system.
            self.springs.extend(new_springs);
            self.leaf_springs.extend(new_leaf_springs);

            // Simulate the system until it reaches stability.
            self.par_simulate_to_stability(dt, patience, target, max_steps);

            // Remove any springs that connect to the parent clusters.
            self.springs.retain(|s| {
                let [a, b] = s.clusters();
                !(parents.contains_key(&a) || parents.contains_key(&b)) || s.k() >= min_k
            });
            self.leaf_springs.retain(|s| {
                let [a, b] = s.clusters();
                !(parents.contains_key(&a) || parents.contains_key(&b)) || s.k() >= min_k
            });

            // Save the current state of the system.
            let path = out_dir.as_ref().join(format!("{}-step-{i}.npy", data.name()));
            self.par_extract_positions().write_npy(&path)?;
        }

        // Simulate the system for longer to ensure that it has reached a stable
        // state with leaf clusters.
        let patience = patience * 10;
        let target = target.map(|t| t * 0.1);
        let max_steps = max_steps.map(|m| m * 10);
        self.par_simulate_to_stability(dt, patience, target, max_steps);

        ftlog::info!(
            "Reached instability of {:.2e} with {} leaf springs after {i} steps.",
            self.instability(patience),
            self.leaf_springs.len()
        );

        Ok(())
    }
}

/// Compute the displacements of the child clusters from the parent cluster. We
/// assume that `c` is the parent cluster and `a` and `b` are the children.
#[allow(clippy::similar_names)]
fn triangle_displacements<T: Number>(ac: T, bc: T, ab: T) -> (f32, f32, f32) {
    // Since the positions are stores as `f32` arrays, we cast the distances
    // to `f32` for internal computations.
    let (fsa, fsb, fab) = (ac.as_f32(), bc.as_f32(), ab.as_f32());

    // Compute the deltas by which to move the child `Mass`es. Note that `a`
    // will only  be moved along the `x` axis while `b` may be moved along
    // one or both axes.
    if crate::utils::is_triangle(ac, bc, ab) {
        // We will move `a` along only the `x` axis and `b` along both axes.

        // Use the law of cosines to compute the length of the projection of
        // `sb` onto the x axis.
        let dxb = (fab.square() - fsa.square() - fsb.square()) / (2.0 * fsa);

        // Use the Pythagorean theorem to compute the length of the
        // projection of `sb` onto the y axis.
        let dyb = (fsb.square() - dxb.square()).sqrt();

        (fsa, dxb, dyb)
    } else {
        // We will move both `a` and `b` along only the `x` axis.
        let (dxa, dxb) = if crate::utils::is_colinear(ac, bc, ab) {
            if ab > ac && ab > bc {
                // `ab` is the longest side `self` in in the middle.
                (-fsa, fsb)
            } else if ac > ab && ac > bc {
                // `sa` is the longest side so `b` is in the middle.
                (fsa + fab, fsb)
            } else {
                // `sb` is the longest side so `a` is in the middle.
                (fsa, fsb + fab)
            }
        } else {
            // The only way that the three distances do not form a triangle
            // is if one of them is larger than the sum of the other two.
            // In this case, we will preserve the two shorter distances and
            // ensure that the largest delta is the sum of the two smaller
            // deltas.

            // Note in the following branches that the longest side does not
            // show up in the returned deltas. This is the only difference
            // between this case and the colinear case.
            if ab > ac && ab > bc {
                // `ab` is the longest side `self` in in the middle.
                (-fsa, fsb)
            } else if ac > ab && ac > bc {
                // `sa` is the longest side so `b` is in the middle.
                (fsb + fab, fsb)
            } else {
                // `sb` is the longest side so `a` is in the middle.
                (fsa, fsa + fab)
            }
        };

        (dxa, dxb, 0.0)
    }
}

impl<const DIM: usize, Me, T: Number, C: Cluster<T>> core::ops::Index<usize> for System<'_, DIM, Me, T, C> {
    type Output = [Vector<DIM>; 2];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const DIM: usize, Me, T: Number, C: Cluster<T>> core::ops::IndexMut<usize> for System<'_, DIM, Me, T, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
