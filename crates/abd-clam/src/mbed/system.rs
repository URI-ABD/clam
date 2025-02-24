//! A wrapper around the `FlatVec` struct that will be used for the mass-spring
//! system.

use std::collections::HashMap;

use distances::{number::Multiplication, Number};
use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    cluster::ParCluster,
    dataset::{AssociatesMetadataMut, ParDataset},
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
#[must_use]
pub struct System<'a, const DIM: usize, Me, T: Number, C: Cluster<T>> {
    /// Each item in the system is a pair of vectors representing the position
    /// and velocity of the point representing the item in the original dataset.
    data: FlatVec<[Vector<DIM>; 2], Me>,
    /// The `Spring`s that connect clusters which have child clusters.
    springs: Vec<Spring<'a, T, C>>,
    /// The `Spring`s that connect leaf clusters.
    leaf_springs: Vec<Spring<'a, T, C>>,
    /// The kinetic energy and potential energy of the system throughout the
    /// simulation.
    energy_history: Vec<(f32, f32)>,
    /// Leaves encountered during the simulation.
    leaves: Vec<&'a C>,
    /// The damping coefficient of the system.
    beta: f32,
    /// The spring constant of the primary springs in the system.
    k: f32,
    /// The factor by which to multiplicatively decrease the spring constant
    /// when loosening springs.
    dk: f32,
    /// The size of the time-step to use in the simulation for each minor step.
    dt: f32,
    /// The fraction of the most displaced springs whose connecting clusters
    /// will be replaced by their children at each major step.
    f: f32,
    /// The number of times a spring will be loosened before being removed.
    retention_depth: usize,
    /// The minimum number of minor steps to wait for the system to reach
    /// stability.
    patience: usize,
    /// The maximum number of minor steps to wait for the system to reach
    /// stability.
    max_steps: usize,
    /// The target instability value to reach before stopping the simulation.
    target: f32,
}

impl<const DIM: usize, T: Number, C: Cluster<T>> System<'_, DIM, usize, T, C> {
    /// Create a new `System` with the given `Cluster` tree.
    ///
    /// # Errors
    ///
    /// - If the data is empty.
    pub fn new(cardinality: usize) -> Result<Self, String> {
        let items = vec![[Vector::zero(); 2]; cardinality];
        let data = FlatVec::new(items)?;
        let springs = Vec::new();
        let leaf_springs = Vec::new();
        let energy_history = vec![(0.0, 0.0)];
        let leaves = Vec::new();

        Ok(Self {
            data,
            springs,
            leaf_springs,
            energy_history,
            leaves,
            beta: 0.99,
            k: 1.0,
            dk: 0.5,
            dt: 1e-3,
            f: 0.5,
            retention_depth: 4,
            patience: 100,
            max_steps: 10_000,
            target: 1e-5,
        })
    }
}

impl<'a, const DIM: usize, Me, T: Number, C: Cluster<T>> System<'a, DIM, Me, T, C> {
    /// Change the metadata of the items in the dataset being simulated.
    ///
    /// # Errors
    ///
    /// - If the number of metadata items does not match the number of items in
    ///   the dataset.
    pub fn with_metadata<Met: Clone>(self, metadata: &[Met]) -> Result<System<'a, DIM, Met, T, C>, String> {
        let data = self.data.with_metadata(metadata)?;
        Ok(System {
            data,
            springs: self.springs,
            leaf_springs: self.leaf_springs,
            energy_history: self.energy_history,
            leaves: self.leaves,
            beta: self.beta,
            k: self.k,
            dk: self.dk,
            dt: self.dt,
            f: self.f,
            retention_depth: self.retention_depth,
            patience: self.patience,
            max_steps: self.max_steps,
            target: self.target,
        })
    }

    /// Sets the initial positions and velocities of the points in the system.
    ///
    /// # Errors
    ///
    /// - If the number of items in the dataset does not match the number of
    ///   items in the system.
    pub fn with_initial_state(mut self, state: &[[Vector<DIM>; 2]]) -> Result<Self, String> {
        if state.len() != self.data.cardinality() {
            let msg = format!(
                "Number of items in the dataset ({}) does not match the number of items in the system ({}).",
                state.len(),
                self.data.cardinality()
            );
            ftlog::error!("{msg}");
            return Err(msg);
        }

        for (i, pv) in state.iter().copied().enumerate() {
            self.data[i] = pv;
        }

        Ok(self)
    }
}

impl<const DIM: usize, Me: Clone, T: Number, C: Cluster<T>> System<'_, DIM, Me, T, C> {
    /// Extract the reduced dataset from the mass-spring system.
    ///
    /// This is a `FlatVec` of `DIM`-dimensional points.
    pub fn extract_positions(&self) -> FlatVec<[f32; DIM], Me> {
        self.data.clone().transform_items(|[p, _]| *p)
    }

    /// Simulate the system until the points reach the leaves of the tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the original items.
    /// - `metric`: The metric to use to calculate the distances between the
    ///   original items.
    /// - `rng`: The random number generator to use in the simulation.
    ///
    /// # Returns
    ///
    /// A vector of `FlatVec`s containing the positions of the points after each
    /// major step of the simulation.
    #[allow(clippy::too_many_lines, clippy::similar_names, clippy::many_single_char_names)]
    pub fn simulate_to_leaves<I, D: Dataset<I>, M: Metric<I, T>, R: rand::Rng>(
        &mut self,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) -> Vec<FlatVec<[f32; DIM], Me>> {
        let mut stressed_centers = vec![false; self.data.cardinality()];
        let mut i = 0;
        let mut steps = Vec::new();
        while !self.springs.is_empty() {
            i += 1;

            // Remove springs that are too weak and sort the remaining springs
            // by their displacement ratio.
            self.remove_weak_springs();
            self.sort_springs_by_displacement();

            // Get the clusters connected to the most displaced springs.
            let at = (self.f * self.springs.len().as_f32()).ceil().as_usize();
            ftlog::debug!(
                "Step {i}, Stopping at {at} stressed springs out of {}.",
                self.springs.len()
            );
            self.springs
                .iter()
                .take(at)
                .flat_map(Spring::clusters)
                .for_each(|c| stressed_centers[c.arg_center()] = true);

            // Remove all springs connected to stressed parents.
            let (connected, disconnected): (Vec<_>, _) = self.springs.drain(..).partition(|s| {
                let [a, b] = s.clusters();
                stressed_centers[a.arg_center()] || stressed_centers[b.arg_center()]
            });
            self.springs = disconnected;
            ftlog::debug!(
                "Step {i}, Removed {} springs connected to stressed centers leaving {} springs.",
                connected.len(),
                self.springs.len()
            );

            // Collect all stressed clusters and their connecting springs.
            let mut parents = HashMap::new();
            for s in connected {
                let ([a, b], _, _, _, num_loosened) = s.deconstruct();
                let num_loosened = num_loosened + 1;
                if !a.is_leaf() {
                    parents.entry(a).or_insert_with(Vec::new).push((b, num_loosened));
                }
                if !b.is_leaf() {
                    parents.entry(b).or_insert_with(Vec::new).push((a, num_loosened));
                }
            }
            ftlog::debug!("Step {i}, Found {} stressed clusters.", parents.len());

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

                    // Choose a pair of random directions for the child clusters.
                    let x = Vector::<DIM>::random_unit(rng);
                    let y = x.perpendicular(rng);

                    // Compute the changes in positions for the child clusters.
                    let pa = self[a.arg_center()][0] + x * dxa;
                    let pb = self[b.arg_center()][0] + x * dxb + y * dyb;

                    // Set the new positions of points in the child clusters.
                    for i in a.indices() {
                        self[i][0] = pa;
                    }
                    for i in b.indices() {
                        self[i][0] = pb;
                    }

                    self.add_spring([a, c], ac, 0);
                    self.add_spring([b, c], bc, 0);

                    let ab = self.new_spring([a, b], ab, 0);
                    if a.is_leaf() && b.is_leaf() {
                        self.leaf_springs.push(ab);
                    } else {
                        self.springs.push(ab);
                    }

                    if a.is_leaf() {
                        self.leaves.push(a);
                    }
                    if b.is_leaf() {
                        self.leaves.push(b);
                    }

                    (c, [a, b])
                })
                .collect::<HashMap<_, _>>();

            // Create springs between the pairs of child clusters of different
            // parent clusters.
            let (new_leaf_springs, new_springs): (Vec<_>, Vec<_>) = triangles
                .iter()
                .flat_map(|(&c, &[ca, cb])| {
                    parents[&c]
                        .iter()
                        // Ensure that springs are only created once.
                        .filter(|&&(d, _)| !d.is_leaf() && c.arg_center() < d.arg_center())
                        .flat_map(|&(d, num_loosened)| {
                            let [da, db] = triangles[&d];
                            [
                                self.new_spring(
                                    [ca, da],
                                    data.one_to_one(ca.arg_center(), da.arg_center(), metric),
                                    num_loosened,
                                ),
                                self.new_spring(
                                    [ca, db],
                                    data.one_to_one(ca.arg_center(), db.arg_center(), metric),
                                    num_loosened,
                                ),
                                self.new_spring(
                                    [cb, da],
                                    data.one_to_one(cb.arg_center(), da.arg_center(), metric),
                                    num_loosened,
                                ),
                                self.new_spring(
                                    [cb, db],
                                    data.one_to_one(cb.arg_center(), db.arg_center(), metric),
                                    num_loosened,
                                ),
                            ]
                        })
                })
                .partition(|s| {
                    let [a, b] = s.clusters();
                    a.is_leaf() && b.is_leaf()
                });
            ftlog::debug!(
                "Step {i}, Created {} new leaf springs and {} new springs.",
                new_leaf_springs.len(),
                new_springs.len()
            );

            // Add the new springs to the system.
            self.springs.extend(new_springs);
            self.leaf_springs.extend(new_leaf_springs);

            // Simulate the system until it reaches stability.
            let instability = self.simulate_to_stability();

            // Remove any springs that connect to the parent clusters.
            self.springs.retain(|s| {
                let [a, b] = s.clusters();
                !(parents.contains_key(&a) || parents.contains_key(&b))
            });
            self.leaf_springs.retain(|s| {
                let [a, b] = s.clusters();
                !(parents.contains_key(&a) || parents.contains_key(&b))
            });

            self.leaf_springs.iter_mut().for_each(|s| {
                s.loosen(self.dk);
            });
            self.add_random_leaf_springs(data, metric, rng);
            stressed_centers = vec![false; self.data.cardinality()];

            ftlog::info!(
                "Step {i}, Finishing with {} springs and {} leaf springs with {instability:.2e} instability.",
                self.springs.len(),
                self.leaf_springs.len()
            );

            // Save the current state of the system.
            steps.push(self.extract_positions());
        }

        let mut instability = self.instability();
        let mut j = 0;
        #[allow(clippy::while_float)]
        while instability > self.target && j < 10 {
            j += 1;
            i += 1;
            instability = self.simulate_to_stability();
            self.leaf_springs.iter_mut().for_each(|s| {
                s.loosen(self.dk);
            });
            self.remove_weak_springs();
            self.add_random_leaf_springs(data, metric, rng);

            ftlog::info!(
                "Step {i}, Finishing with {} springs and {} leaf springs with {instability:.2e} instability.",
                self.springs.len(),
                self.leaf_springs.len()
            );

            // Save the current state of the system.
            steps.push(self.extract_positions());
        }

        ftlog::info!(
            "Reached instability of {:.2e} with {} leaf clusters after {i} steps.",
            self.instability(),
            self.leaves.len()
        );

        steps
    }
}

impl<const DIM: usize, Me: Clone + Send + Sync, T: Number, C: Cluster<T>> System<'_, DIM, Me, T, C> {
    /// Parallel version of [`extract_positions`](Self::extract_positions).
    pub fn par_extract_positions(&self) -> FlatVec<[f32; DIM], Me> {
        self.data.clone().par_transform_items(|[p, _]| *p)
    }
}

impl<const DIM: usize, Me: Clone + Send + Sync, T: Number, C: ParCluster<T>> System<'_, DIM, Me, T, C> {
    /// Parallel version of [`simulate_to_leaves`](Self::simulate_to_leaves).
    #[allow(clippy::too_many_lines, clippy::similar_names, clippy::many_single_char_names)]
    pub fn par_simulate_to_leaves<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>, R: rand::Rng>(
        &mut self,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) -> Vec<FlatVec<[f32; DIM], Me>> {
        let mut stressed_centers = vec![false; self.data.cardinality()];
        let mut i = 0;
        let mut steps = Vec::new();
        while !self.springs.is_empty() {
            i += 1;

            // Remove springs that are too weak and sort the remaining springs
            // by their displacement ratio.
            self.remove_weak_springs();
            self.par_sort_springs_by_displacement();

            // Get the clusters connected to the most displaced springs.
            let at = (self.f * self.springs.len().as_f32()).ceil().as_usize();
            ftlog::debug!(
                "Step {i}, Stopping at {at} stressed springs out of {}.",
                self.springs.len()
            );
            self.springs
                .iter()
                .take(at)
                .flat_map(Spring::clusters)
                .for_each(|c| stressed_centers[c.arg_center()] = true);

            // Remove all springs connected to stressed parents.
            let (connected, disconnected): (Vec<_>, _) = self.springs.par_drain(..).partition(|s| {
                let [a, b] = s.clusters();
                stressed_centers[a.arg_center()] || stressed_centers[b.arg_center()]
            });
            self.springs = disconnected;
            ftlog::debug!(
                "Step {i}, Removed {} springs connected to stressed centers leaving {} springs.",
                connected.len(),
                self.springs.len()
            );

            // Collect all stressed clusters and their connecting springs.
            let mut parents = HashMap::new();
            for s in connected {
                let ([a, b], _, _, _, num_loosened) = s.deconstruct();
                let num_loosened = num_loosened + 1;
                if !a.is_leaf() {
                    parents.entry(a).or_insert_with(Vec::new).push((b, num_loosened));
                }
                if !b.is_leaf() {
                    parents.entry(b).or_insert_with(Vec::new).push((a, num_loosened));
                }
            }
            ftlog::debug!("Step {i}, Found {} stressed clusters.", parents.len());

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
                    // Choose a pair of random directions for the child clusters.
                    let x = Vector::<DIM>::random_unit(rng);
                    let y = x.perpendicular(rng);

                    // Compute the changes in positions for the child clusters.
                    let pa = self[a.arg_center()][0] + x * dxa;
                    let pb = self[b.arg_center()][0] + x * dxb + y * dyb;

                    // Set the new positions of points in the child clusters.
                    for i in a.indices() {
                        self[i][0] = pa;
                    }
                    for i in b.indices() {
                        self[i][0] = pb;
                    }

                    self.add_spring([a, c], ac, 0);
                    self.add_spring([b, c], bc, 0);

                    let ab = self.new_spring([a, b], ab, 0);
                    if a.is_leaf() && b.is_leaf() {
                        self.leaf_springs.push(ab);
                    } else {
                        self.springs.push(ab);
                    }

                    if a.is_leaf() {
                        self.leaves.push(a);
                    }
                    if b.is_leaf() {
                        self.leaves.push(b);
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
                        // Ensure that springs are only created once.
                        .filter(|&&(d, _)| !d.is_leaf() && c.arg_center() < d.arg_center())
                        .flat_map(|&(d, num_loosened)| {
                            let [da, db] = triangles[&d];
                            [
                                self.new_spring(
                                    [ca, da],
                                    data.one_to_one(ca.arg_center(), da.arg_center(), metric),
                                    num_loosened,
                                ),
                                self.new_spring(
                                    [ca, db],
                                    data.one_to_one(ca.arg_center(), db.arg_center(), metric),
                                    num_loosened,
                                ),
                                self.new_spring(
                                    [cb, da],
                                    data.one_to_one(cb.arg_center(), da.arg_center(), metric),
                                    num_loosened,
                                ),
                                self.new_spring(
                                    [cb, db],
                                    data.one_to_one(cb.arg_center(), db.arg_center(), metric),
                                    num_loosened,
                                ),
                            ]
                        })
                })
                .partition(|s| {
                    let [a, b] = s.clusters();
                    a.is_leaf() && b.is_leaf()
                });
            ftlog::debug!(
                "Step {i}, Created {} new leaf springs and {} new springs.",
                new_leaf_springs.len(),
                new_springs.len()
            );

            // Add the new springs to the system.
            self.springs.extend(new_springs);
            self.leaf_springs.extend(new_leaf_springs);

            // Simulate the system until it reaches stability.
            let instability = self.par_simulate_to_stability();

            // Remove any springs that connect to the parent clusters.
            self.springs.retain(|s| {
                let [a, b] = s.clusters();
                !(parents.contains_key(&a) || parents.contains_key(&b))
            });
            self.leaf_springs.retain(|s| {
                let [a, b] = s.clusters();
                !(parents.contains_key(&a) || parents.contains_key(&b))
            });

            // Decrease the spring constants of the leaf springs.
            self.leaf_springs.par_iter_mut().for_each(|s| {
                s.loosen(self.dk);
            });
            // Add new random leaf springs.
            self.par_add_random_leaf_springs(data, metric, rng);
            // Reset the stressed centers.
            stressed_centers = vec![false; self.data.cardinality()];

            ftlog::info!(
                "Step {i}, Finishing with {} springs and {} leaf springs with {instability:.2e} instability.",
                self.springs.len(),
                self.leaf_springs.len()
            );

            // Save the current state of the system.
            steps.push(self.par_extract_positions());
        }

        let mut instability = self.instability();
        let mut j = 0;
        #[allow(clippy::while_float)]
        while instability > self.target && j < 10 {
            j += 1;
            i += 1;
            instability = self.par_simulate_to_stability();
            self.leaf_springs.par_iter_mut().for_each(|s| {
                s.loosen(self.dk);
            });
            self.remove_weak_springs();
            self.par_add_random_leaf_springs(data, metric, rng);

            ftlog::info!(
                "Step {i}, Finishing with {} springs and {} leaf springs with {instability:.2e} instability.",
                self.springs.len(),
                self.leaf_springs.len()
            );

            // Save the current state of the system.
            steps.push(self.par_extract_positions());
        }

        ftlog::info!(
            "Reached instability of {:.2e} with {} leaf clusters after {i} steps.",
            self.instability(),
            self.leaves.len()
        );

        steps
    }
}

impl<'a, const DIM: usize, Me, T: Number, C: Cluster<T>> System<'a, DIM, Me, T, C> {
    /// Return the damping coefficient of the system. The default is `0.99`.
    #[must_use]
    pub const fn beta(&self) -> f32 {
        self.beta
    }

    /// Change the damping coefficient of the system. The default is `0.99`.
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

    /// Return the spring constant of the primary springs in the system. The
    /// default is `1.0`.
    #[must_use]
    pub const fn k(&self) -> f32 {
        self.k
    }

    /// Change the spring constant of the primary springs in the system. The
    /// default is `1.0`.
    ///
    /// # Errors
    ///
    /// - If `k` is not positive.
    pub fn with_k(mut self, k: f32) -> Result<Self, String> {
        if k <= 0.0 {
            let msg = format!("Spring constant must be positive. Got {k:.2e} instead.");
            ftlog::error!("{msg}");
            Err(msg)
        } else {
            self.k = k;
            Ok(self)
        }
    }

    /// Return the factor by which to multiplicatively decrease the spring
    /// constant when loosening springs.
    #[must_use]
    pub const fn dk(&self) -> f32 {
        self.dk
    }

    /// Change the factor by which to multiplicatively decrease the spring
    /// constant when loosening springs. The default is `0.5`.
    ///
    /// # Errors
    ///
    /// - If `dk` is not in the range `(0, 1)`.
    pub fn with_dk(mut self, dk: f32) -> Result<Self, String> {
        if dk <= 0.0 || dk >= 1.0 {
            let msg = format!("`dk` must be in the range (0, 1). Got {dk:.2e} instead.");
            ftlog::error!("{msg}");
            Err(msg)
        } else {
            self.dk = dk;
            Ok(self)
        }
    }

    /// Return the size of the time-step to use in the simulation for each minor
    /// step. The default is `1e-3`.
    #[must_use]
    pub const fn dt(&self) -> f32 {
        self.dt
    }

    /// Change the size of the time-step to use in the simulation for each minor
    /// step. The default is `1e-3`.
    ///
    /// # Errors
    ///
    /// - If `dt` is not positive.
    pub fn with_dt(mut self, dt: f32) -> Result<Self, String> {
        if dt <= 0.0 {
            let msg = format!("Time-step must be positive. Got {dt:.2e} instead.");
            ftlog::error!("{msg}");
            Err(msg)
        } else {
            self.dt = dt;
            Ok(self)
        }
    }

    /// Return the fraction of the most displaced springs whose connecting
    /// clusters will be replaced by their children at each major step. The
    /// default is `0.5`.
    #[must_use]
    pub const fn f(&self) -> f32 {
        self.f
    }

    /// Change the fraction of the most displaced springs whose connecting
    /// clusters will be replaced by their children at each major step. The
    /// default is `0.5`.
    ///
    /// # Errors
    ///
    /// - If `f` is not in the range `(0, 1)`.
    pub fn with_f(mut self, f: f32) -> Result<Self, String> {
        if f <= 0.0 || f >= 1.0 {
            let msg = format!("`f` must be in the range (0, 1). Got {f:.2e} instead.");
            ftlog::error!("{msg}");
            Err(msg)
        } else {
            self.f = f;
            Ok(self)
        }
    }

    /// Return the number of times a spring will be loosened before being
    /// removed. The default is `4`.
    #[must_use]
    pub const fn retention_depth(&self) -> usize {
        self.retention_depth
    }

    /// Change the number of times a spring will be loosened before being
    /// removed. The default is `4`.
    pub const fn with_retention_depth(mut self, retention_depth: usize) -> Self {
        self.retention_depth = retention_depth;
        self
    }

    /// Return the minimum number of minor steps to wait for the system to reach
    /// stability. The default is `100`.
    #[must_use]
    pub const fn patience(&self) -> usize {
        self.patience
    }

    /// Change the minimum number of minor steps to wait for the system to reach
    /// stability. The default is `100`.
    pub const fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Return the maximum number of minor steps to wait for the system to reach
    /// stability. The default is `10_000`.
    #[must_use]
    pub const fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Change the maximum number of minor steps to wait for the system to reach
    /// stability. The default is `10_000`.
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Return the target instability value to reach before stopping the
    /// simulation. The default is `1e-5`.
    #[must_use]
    pub const fn target(&self) -> f32 {
        self.target
    }

    /// Change the target instability value to reach before stopping the
    /// simulation. The default is `1e-5`.
    ///
    /// # Errors
    ///
    /// - If `target` is not positive.
    pub fn with_target(mut self, target: f32) -> Result<Self, String> {
        if target <= 0.0 {
            let msg = format!("Target instability must be positive. Got {target:.2e} instead.");
            ftlog::error!("{msg}");
            Err(msg)
        } else {
            self.target = target;
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
        root: &'a C,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) {
        let children = root.children();
        if children.len() < 2 {
            let msg = "Root cluster must have at least two children.";
            ftlog::error!("{msg}");
            return;
        }

        // For each child, choose a random position inside a hypercube centered
        // at the origin with side length equal to the diameter of the root.
        let radius = root.radius().as_f32();
        for &c in &children {
            let p = Vector::random(rng, -radius, radius);
            let v = Vector::random(rng, -1.0, 1.0);
            for i in c.indices() {
                self[i] = [p, v];
            }

            // If the child is a leaf, add it to the list of leaves.
            if c.is_leaf() {
                self.leaves.push(c);
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
                            Some(Spring::new([a, b], l0, l, 0, self.dk))
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
    }

    /// Adds springs among random pairs of leaf clusters.
    fn add_random_leaf_springs<I, D: Dataset<I>, M: Metric<I, T>, R: rand::Rng>(
        &mut self,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) {
        self.leaves.shuffle(rng);
        let mut new_springs = self
            .leaves
            .iter()
            .zip(self.leaves.iter().rev())
            .filter_map(|(&a, &b)| {
                if a.arg_center() < b.arg_center() {
                    let l0 = data.one_to_one(a.arg_center(), b.arg_center(), metric);
                    Some(self.new_spring([a, b], l0, 0))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        self.leaf_springs.append(&mut new_springs);
    }

    /// Adds a new spring between the given clusters.
    pub fn add_spring(&mut self, [a, b]: [&'a C; 2], l0: T, num_loosened: usize) {
        let spring = self.new_spring([a, b], l0, num_loosened);
        if a.is_leaf() && b.is_leaf() {
            self.leaf_springs.push(spring);
        } else {
            self.springs.push(spring);
        }
    }

    /// Creates a new spring between the given clusters.
    pub fn new_spring(&self, [a, b]: [&'a C; 2], l0: T, num_loosened: usize) -> Spring<'a, T, C> {
        let l = self.distance_between(a.arg_center(), b.arg_center());
        Spring::new([a, b], l0, l, num_loosened, self.dk)
    }

    /// Remove springs and leaf-springs whose spring constant is below the given
    /// threshold.
    fn remove_weak_springs(&mut self) {
        self.springs.retain(|s| s.is_intact(self.retention_depth));
        self.leaf_springs.retain(|s| s.is_intact(self.retention_depth));
    }

    /// Sort the springs by their displacement ratio in descending order.
    fn sort_springs_by_displacement(&mut self) {
        self.springs.sort_by(|a, b| b.ratio().total_cmp(&a.ratio()));
        self.leaf_springs.sort_by(|a, b| b.ratio().total_cmp(&a.ratio()));
    }

    /// Get the total kinetic energy of the system.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.springs
            .iter()
            .chain(self.leaf_springs.iter())
            .flat_map(Spring::clusters)
            .map(|c| self.data[c.arg_center()][1].magnitude().square() * c.cardinality().as_f32())
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

    /// Get the instability of the `System` over the last `patience` time-steps.
    ///
    /// The instability is the sum of the mean kinetic energy of all masses and
    /// the coefficient of variation of the potential energy of all springs.
    ///
    /// # Returns
    ///
    /// The instability of the `System` in a [0, inf) range, lower values
    /// indicating higher stability.
    #[must_use]
    pub fn instability(&'a self) -> f32 {
        if self.energy_history.len() < self.patience {
            1.0
        } else {
            let (ke_values, pe_values): (Vec<_>, Vec<_>) = self
                .energy_history
                .iter()
                .skip(self.energy_history.len() - self.patience)
                .copied()
                .unzip();

            // let var_kinetic: f32 = crate::utils::coefficient_of_variation(&ke_values);
            let mean_kinetic: f32 = crate::utils::mean(&ke_values);
            let var_potential: f32 = crate::utils::coefficient_of_variation(&pe_values);

            mean_kinetic + var_potential
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
            .chain(self.leaf_springs.iter())
            .map(|s| {
                let [a, b] = s.clusters();
                self.distance_between(a.arg_center(), b.arg_center())
            })
            .collect::<Vec<_>>();
        self.springs
            .iter_mut()
            .chain(self.leaf_springs.iter_mut())
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
    pub fn update_step(&mut self) {
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
        let accelerations = {
            let mut accelerations = (0..self.data.cardinality())
                .map(|i| self[i][1] * self.beta)
                .collect::<Vec<_>>();
            for (c, f) in forces {
                let m = c.cardinality().as_f32();
                let a = f / m;
                for i in c.indices() {
                    // The addition here accounts for the accumulated forces.
                    accelerations[i] += a;
                }
            }

            accelerations
        };

        // Apply the changes to the positions and velocities of the points.
        self.data.transform_items_enumerated_in_place(|i, [mut x, mut v]| {
            v += accelerations[i] * self.dt;
            x += v * self.dt;
        });

        // Update the springs in the system.
        self.update_springs();

        // Update the energy history of the system.
        self.update_energy_history();
    }

    /// Simulate the system until it reaches a stable state.
    #[must_use]
    pub fn simulate_to_stability(&mut self) -> f32 {
        let mut i = 0;
        let mut instability = self.instability();
        while i < self.patience || (instability > self.target && i < self.max_steps) {
            i += 1;
            self.update_step();
            instability = self.instability();
        }

        ftlog::info!(
            "Reached instability of {instability:.2e} after {i} steps with {} springs.",
            self.springs.len()
        );

        instability
    }
}

impl<'a, const DIM: usize, Me: Send + Sync, T: Number, C: ParCluster<T>> System<'a, DIM, Me, T, C> {
    /// Parallel version of [`kinetic_energy`](Self::kinetic_energy).
    #[must_use]
    pub fn par_kinetic_energy(&self) -> f32 {
        self.springs
            .par_iter()
            .chain(self.leaf_springs.par_iter())
            .flat_map(Spring::clusters)
            .map(|c| self.data[c.arg_center()][1].magnitude().square() * c.cardinality().as_f32())
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
    pub fn par_initialize_with_root<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>, R: rand::Rng>(
        &mut self,
        root: &'a C,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) {
        let children = root.children();
        if children.len() < 2 {
            let msg = "Root cluster must have at least two children.";
            ftlog::error!("{msg}");
            return;
        }

        // For each child, choose a random position inside a hypercube centered
        // at the origin with side length equal to the diameter of the root.
        let radius = root.radius().as_f32();
        for &c in &children {
            let p = Vector::random(rng, -radius, radius);
            let v = Vector::random(rng, -1.0, 1.0);
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
                            Some(Spring::new([a, b], l0, l, 0, self.dk))
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
    }

    /// Parallel version of [`update_springs`](Self::update_springs).
    fn par_update_springs(&mut self) {
        let new_lengths = self
            .springs
            .par_iter()
            .chain(self.leaf_springs.par_iter())
            .map(|s| {
                let [a, b] = s.clusters();
                self.distance_between(a.arg_center(), b.arg_center())
            })
            .collect::<Vec<_>>();
        self.springs
            .par_iter_mut()
            .chain(self.leaf_springs.par_iter_mut())
            .zip(new_lengths)
            .for_each(|(s, l)| s.update_length(l));
    }

    /// Sort the springs by their displacement ratio in descending order.
    fn par_sort_springs_by_displacement(&mut self) {
        self.springs.par_sort_by(|a, b| b.ratio().total_cmp(&a.ratio()));
        self.leaf_springs.par_sort_by(|a, b| b.ratio().total_cmp(&a.ratio()));
    }

    /// Parallel version of [`update_step`](Self::update_step).
    pub fn par_update_step(&mut self) {
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
        let accelerations = {
            let mut accelerations = (0..self.data.cardinality())
                .map(|i| self[i][1] * self.beta)
                .collect::<Vec<_>>();
            for (c, f) in forces {
                let m = c.cardinality().as_f32();
                let a = f / m;
                for i in c.indices() {
                    // The addition here accounts for the accumulated forces.
                    accelerations[i] += a;
                }
            }

            accelerations
        };

        // Apply the changes to the positions and velocities of the points.
        self.data.par_transform_items_enumerated_in_place(|i, [mut x, mut v]| {
            v += accelerations[i] * self.dt;
            x += v * self.dt;
        });

        // Update the springs in the system.
        self.par_update_springs();

        // Update the energy history of the system.
        self.par_update_energy_history();
    }

    /// Parallel version of [`simulate_to_stability`](Self::simulate_to_stability).
    #[must_use]
    pub fn par_simulate_to_stability(&mut self) -> f32 {
        let mut i = 0;
        let mut instability = self.instability();
        while i < self.patience || (instability > self.target && i < self.max_steps) {
            i += 1;
            self.par_update_step();
            instability = self.instability();
        }

        ftlog::debug!(
            "Reached instability of {instability:.2e} after {i} steps with {} springs and {} leaf springs.",
            self.springs.len(),
            self.leaf_springs.len()
        );

        instability
    }

    /// Adds springs among random pairs of leaf clusters.
    fn par_add_random_leaf_springs<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>, R: rand::Rng>(
        &mut self,
        data: &D,
        metric: &M,
        rng: &mut R,
    ) {
        self.leaves.shuffle(rng);
        let mut new_springs = self
            .leaves
            .par_iter()
            .zip(self.leaves.par_iter().rev())
            .filter_map(|(&a, &b)| {
                if a.arg_center() < b.arg_center() {
                    let l0 = data.one_to_one(a.arg_center(), b.arg_center(), metric);
                    Some(self.new_spring([a, b], l0, 0))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        self.leaf_springs.append(&mut new_springs);
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
