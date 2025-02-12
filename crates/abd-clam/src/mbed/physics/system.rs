//! A mass-spring system for dimension reduction.

use std::collections::{hash_map::Entry, HashMap};

use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    chaoda::Graph, cluster::ParCluster, core::adapters::Adapted, dataset::ParDataset, metric::ParMetric, Cluster,
    Dataset, FlatVec, Metric, SizedHeap,
};

use super::{Mass, Spring};

/// A `HashMap` of `Mass`es from their hash keys.
pub type Masses<'a, const DIM: usize, T, S> = HashMap<(usize, usize), Mass<'a, DIM, T, S>>;

/// A mass-spring system for dimension reduction.
#[must_use]
pub struct System<'a, const DIM: usize, T: Number, S: Cluster<T>> {
    /// The masses in the system.
    masses: Masses<'a, DIM, T, S>,
    /// The springs in the system.
    springs: Vec<Spring>,
    /// The damping factor of the system.
    beta: f32,
    /// The energy values of the system as it evolved over time.
    energies: Vec<[f32; 3]>,
    /// The name of the system.
    name: String,
}

impl<'a, const DIM: usize, T: Number, S: Cluster<T>> System<'a, DIM, T, S> {
    /// Creates a new `System` of a single `Mass` from a root `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `root`: The root `Cluster` to create the `System` from.
    /// - `beta`: The damping factor of the `System`.
    /// - `name`: The name of the `System`.
    pub fn from_root(root: &'a S, beta: f32, name: &str) -> Self {
        let m = Mass::new(root);
        let m_key = m.hash_key();
        let masses = core::iter::once((m_key, m)).collect::<HashMap<_, _>>();
        Self {
            masses,
            springs: Vec::new(),
            beta,
            energies: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Iteratively replaces the most stressed non-leaf `Mass` with its children
    /// and evolves the `System` until it reaches a stable state with only leaf
    /// `Mass`es.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset on which the `Cluster` tree was built.
    /// - `metric`: The metric to use for distance calculations.
    /// - `seed`: The seed for the random number generator.
    /// - `k`: The spring constant of the first-order `Springs`.
    /// - `f`: The factor by which to multiply the spring constant of the
    ///   inherited springs. This will be clamped to the [0, 1] range.
    /// - `min_k`: The minimum spring constant to consider for removing springs.
    /// - `dt`: The time-step for the simulation.
    /// - `patience`: The number of time-steps to consider for stability.
    /// - `target`: The target stability to reach.
    /// - `max_steps`: The maximum number of time-steps to simulate per
    ///   iteration.
    #[allow(clippy::too_many_arguments)]
    pub fn evolve_to_leaves<I, D: Dataset<I>, M: Metric<I, T>, P: AsRef<std::path::Path>>(
        mut self,
        data: &D,
        metric: &M,
        seed: Option<u64>,
        k: f32,
        f: f32,
        min_k: Option<f32>,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
        out_dir: &P,
    ) -> Self {
        let f = f.clamp(0.0, 1.0);

        let mut m_keys = self
            .most_stressed_non_leaves(100)
            .into_iter()
            .map(Mass::hash_key)
            .collect::<Vec<_>>();
        if let Some(min_k) = min_k {
            while !m_keys.is_empty() {
                self = m_keys.iter().fold(self, |system, m_key| {
                    system.add_child_triangle(*m_key, data, metric, seed, k, f).0
                });
                self = self
                    .remove_loose_springs(min_k)
                    .0
                    .add_random_springs(data, metric, k)
                    .evolve_to_stability(dt, patience, target, max_steps);
                self = m_keys.iter().fold(self, |system, &m_key| system.remove_mass(m_key).0);
                self = self.clear_energies();
                m_keys = self
                    .most_stressed_non_leaves(100)
                    .into_iter()
                    .map(Mass::hash_key)
                    .collect::<Vec<_>>();

                let path = out_dir
                    .as_ref()
                    .join(format!("{}-step-{}.npy", self.name, self.masses.len()));
                self.get_reduced_embedding()
                    .write_npy(&path)
                    .unwrap_or_else(|e| unreachable!("Error writing embeddings: {e}"));
            }
        } else {
            while !m_keys.is_empty() {
                self = m_keys.iter().fold(self, |system, m_key| {
                    system.add_child_triangle(*m_key, data, metric, seed, k, f).0
                });
                self = self
                    .add_random_springs(data, metric, k)
                    .evolve_to_stability(dt, patience, target, max_steps);
                self = m_keys.iter().fold(self, |system, &m_key| system.remove_mass(m_key).0);
                self = self.clear_energies();
                m_keys = self
                    .most_stressed_non_leaves(100)
                    .into_iter()
                    .map(Mass::hash_key)
                    .collect::<Vec<_>>();

                let path = out_dir
                    .as_ref()
                    .join(format!("{}-step-{}.npy", self.name, self.masses.len()));
                self.get_reduced_embedding()
                    .write_npy(&path)
                    .unwrap_or_else(|e| unreachable!("Error writing embeddings: {e}"));
            }
        }

        self.evolve_to_stability(dt, patience * 10, target, None)
    }

    /// Creates a new `System` of `Mass`es from a `Graph`.
    ///
    /// # Arguments
    ///
    /// - `g`: The `Graph` to create the `System` from.
    /// - `beta`: The damping factor of the `System`.
    /// - `k`: The spring constant of the `Springs`.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the points in the original embedding space.
    /// - `U`: The type of the distance values.
    /// - `D`: The dataset.
    /// - `C`: The type of the `Cluster`s in the `Graph`.
    pub fn from_graph(g: &'a Graph<T, S>, beta: f32, k: f32, name: &str) -> Self {
        let masses = g
            .iter_clusters()
            .map(Adapted::source)
            .map(Mass::new)
            .map(|m| (m.hash_key(), m))
            .collect::<HashMap<_, _>>();
        let springs = g
            .iter_edges()
            .map(|(a, b, l0)| {
                let (a_key, b_key) = (a.unique_id(), b.unique_id());
                let l = masses[&a_key].current_distance_to(&masses[&b_key]);
                Spring::new(a_key, b_key, k, l0, l)
            })
            .collect();
        Self {
            masses,
            springs,
            beta,
            energies: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Sets random positions for all masses.
    ///
    /// The positions will be inside a cube with side length `side_length`
    /// centered at the origin.
    ///
    /// # Arguments
    ///
    /// - `side_length`: The side length of the cube.
    /// - `seed`: The seed for the random number generator.
    pub fn init_random(mut self, side_length: f32, seed: Option<u64>) -> Self {
        let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

        let (min_, max_) = (-side_length / 2.0, side_length / 2.0);
        for m in self.masses.values_mut() {
            let mut position = [0.0; DIM];
            for p in &mut position {
                *p = rng.gen_range(min_..max_);
            }
            m.set_position(position);
        }

        self
    }

    /// Returns the masses in the system.
    #[must_use]
    pub const fn masses(&self) -> &Masses<DIM, T, S> {
        &self.masses
    }

    /// Returns the springs in the system.
    pub fn springs(&self) -> &[Spring] {
        &self.springs
    }

    /// Returns the damping factor of the system.
    #[must_use]
    pub const fn beta(&self) -> f32 {
        self.beta
    }

    /// Returns the energy values of the system as it evolved over time.
    #[must_use]
    pub fn energies(&self) -> &[[f32; 3]] {
        &self.energies
    }

    /// Returns the name of the system.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Changes the name of the system.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Updates the lengths and forces of the springs in the system.
    pub fn update_springs(mut self) -> Self {
        self.springs.iter_mut().for_each(|s| {
            let l = self.masses[&s.a_key()].current_distance_to(&self.masses[&s.b_key()]);
            s.update_length(l);
        });
        self
    }

    /// Clears the energy values of the system.
    pub fn clear_energies(mut self) -> Self {
        self.energies.clear();
        self
    }

    /// Updates the `System` for one time step.
    ///
    /// This will:
    /// - calculate the forces exerted by the springs on the masses.
    /// - apply the forces to the masses to move them.
    /// - update the lengths and forces of the springs.
    /// - calculate the energy of the system.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time step to update the system for.
    pub fn update_step(mut self, dt: f32) -> Self {
        let forces = self
            .springs
            .iter()
            .map(|s| {
                let (a_key, b_key) = s.hash_key();
                let f_mag = s.f_mag();
                let mut fv = self.masses[&a_key].unit_vector_to(&self.masses[&b_key]);
                for f_i in &mut fv {
                    *f_i *= f_mag;
                }
                (a_key, b_key, fv)
            })
            .collect::<Vec<_>>();

        for (a_key, b_key, f) in forces {
            if let Some(m) = self.masses.get_mut(&a_key) {
                m.add_force(f);
            }
            if let Some(m) = self.masses.get_mut(&b_key) {
                m.sub_force(f);
            }
        }

        self.masses.iter_mut().for_each(|(_, m)| m.apply_force(dt, self.beta));

        self.energies.push(self.current_energies());

        self.update_springs()
    }

    /// Simulate the `System` until it reaches a stable state.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `patience`: The number of time-steps to consider for stability.
    /// - `target`: The target stability to reach.
    /// - `max_steps`: The maximum number of time-steps to simulate.
    pub fn evolve_to_stability(
        mut self,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
    ) -> Self {
        let target = target.unwrap_or(0.995);
        let max_steps = max_steps.unwrap_or(usize::MAX);

        self.energies.push(self.current_energies());

        let mut i = 0;
        let mut stability = self.stability(patience);
        while stability > target && i < max_steps {
            self = self.update_step(dt);
            i += 1;
            stability = self.stability(patience);
        }

        ftlog::debug!(
            "Reached stability: {stability:.10} after {i} steps with {} objects and {} springs",
            self.masses.len(),
            self.springs.len()
        );

        self
    }

    /// Simulate the `System` for a given number of time-steps.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `steps`: The number of time-steps to simulate.
    pub fn evolve(mut self, dt: f32, steps: usize) -> Self {
        self.energies.push(self.current_energies());

        for i in 0..steps {
            ftlog::debug!("Step {}/{steps}", i + 1);
            self = self.update_step(dt);
        }

        self
    }

    /// Get the dimension-reduced embedding from this `System`.
    ///
    /// This will return a `FlatVec` with the following properties:
    ///
    /// - The instances are the positions of the `Mass`es.
    /// - The metadata are the indices of the centers of the `Cluster`s
    ///   represented by the `Mass`es.
    /// - The distance function is the Euclidean distance.
    pub fn get_reduced_embedding(&self) -> FlatVec<[f32; DIM], usize> {
        let mut positions = self
            .masses
            .values()
            .flat_map(|m| m.indices().iter().map(move |&i| (i, m.position())).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        positions.sort_by(|(a, _), (b, _)| a.cmp(b));

        let positions = positions.into_iter().map(|(_, &p)| p).collect::<Vec<_>>();
        FlatVec::from_arrays(positions).unwrap_or_else(|e| unreachable!("Error creating FlatVec: {e}"))
    }

    /// Simulate the `System` until reaching the target stability, saving the
    /// reduced embedding at regular intervals.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `patience`: The number of time-steps to consider for stability.
    /// - `target`: The target stability to reach.
    /// - `max_steps`: The maximum number of time-steps to simulate.
    /// - `save_every`: The number of time-steps between saves.
    /// - `dir`: The directory to save the embeddings to.
    /// - `name`: The name of the embeddings.
    ///
    /// # Errors
    ///
    /// If there is an error writing the embeddings to disk.
    #[cfg(feature = "disk-io")]
    #[allow(clippy::too_many_arguments)]
    pub fn evolve_to_stability_with_saves<P: AsRef<std::path::Path>>(
        mut self,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
        save_every: usize,
        dir: P,
        name: &str,
    ) -> Result<Self, String> {
        let target = target.unwrap_or(0.995);
        let max_steps = max_steps.unwrap_or(usize::MAX);
        self.energies.push(self.current_energies());

        let mut i = 0;
        let mut stability = self.stability(patience);
        while stability.is_nan() || (stability < target && i < max_steps) {
            if i % save_every == 0 {
                ftlog::debug!("{name}: Saving step {}", i + 1);
                let path = dir.as_ref().join(format!("{}.npy", i + 1));
                self.get_reduced_embedding().write_npy(&path)?;
            }

            ftlog::debug!(
                "Reached stability: {stability:.6} after {i} steps with {} objects and {} springs",
                self.masses.len(),
                self.springs.len()
            );
            self = self.update_step(dt);
            i += 1;
            stability = self.stability(patience);
        }

        ftlog::debug!(
            "Reached stability: {stability:.6} after {i} steps with {} objects and {} springs",
            self.masses.len(),
            self.springs.len()
        );

        Ok(self)
    }

    /// Simulate the `System` for a given number of time-steps, saving the
    /// reduced embedding at regular intervals.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `steps`: The number of time-steps to simulate.
    /// - `save_every`: The number of time-steps between saves.
    /// - `dir`: The directory to save the embeddings to.
    /// - `name`: The name of the embeddings.
    ///
    /// # Errors
    ///
    /// If there is an error writing the embeddings to disk.
    #[cfg(feature = "disk-io")]
    pub fn evolve_with_saves<P: AsRef<std::path::Path>>(
        mut self,
        dt: f32,
        steps: usize,
        save_every: usize,
        dir: P,
        name: &str,
    ) -> Result<Self, String> {
        self.energies.push(self.current_energies());

        for i in 0..steps {
            if i % save_every == 0 {
                ftlog::debug!("{name}: Saving step {}/{steps}", i + 1);
                let path = dir.as_ref().join(format!("{}.npy", i + 1));
                self.get_reduced_embedding().write_npy(&path)?;
            }
            self = self.update_step(dt);
        }

        Ok(self)
    }

    /// Get the total kinetic energy of the `System`.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.masses.values().map(Mass::kinetic_energy).sum()
    }

    /// Get the total potential energy of the `System`.
    #[must_use]
    pub fn potential_energy(&self) -> f32 {
        self.springs.iter().map(Spring::potential_energy).sum()
    }

    /// Get the total energy of the `System`.
    #[must_use]
    pub fn total_energy(&self) -> f32 {
        self.potential_energy() + self.kinetic_energy()
    }

    /// Calculates the energy of the system.
    #[must_use]
    pub fn current_energies(&self) -> [f32; 3] {
        let kinetic_energy = self.kinetic_energy();
        let potential_energy = self.potential_energy();
        let total_energy = kinetic_energy + potential_energy;
        [kinetic_energy, potential_energy, total_energy]
    }

    /// Get the stability of the `System` over the last `n` time-steps.
    ///
    /// The stability is the mean of the coefficient of variation of the kinetic
    /// and potential energies over the last `n` time-steps.
    ///
    /// # Arguments
    ///
    /// - `n`: The number of time-steps to consider.
    ///
    /// # Returns
    ///
    /// The stability of the `System` in a [0, inf) range, lower values
    /// indicating higher stability.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn stability(&'a self, n: usize) -> f32 {
        if self.energies.len() < n || self.masses.len() < 2 || self.springs.is_empty() {
            1.0
        } else {
            let (last_ke, last_pe) = self
                .energies
                .iter()
                .skip(self.energies.len() - n)
                .map(|&[ke, pe, _]| (ke, pe))
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let rel_var_ke = {
                let mean = crate::utils::mean::<_, f32>(&last_ke);
                let variance = crate::utils::variance(&last_ke, mean);
                if mean <= f32::EPSILON {
                    variance.sqrt()
                } else {
                    variance.sqrt() / mean
                }
            };

            let rel_var_pe = {
                let mean = crate::utils::mean::<_, f32>(&last_pe);
                let variance = crate::utils::variance(&last_pe, mean);
                if mean <= f32::EPSILON {
                    variance.sqrt()
                } else {
                    variance.sqrt() / mean
                }
            };

            (rel_var_ke + rel_var_pe) / 2.0
        }
    }

    /// Adds a `Mass` to the `System`.
    ///
    /// # Errors
    ///
    /// - If the `Mass` already exists in the `System`.
    pub fn with_mass(mut self, m: Mass<'a, DIM, T, S>) -> Result<Self, String> {
        if let Entry::Vacant(e) = self.masses.entry(m.hash_key()) {
            e.insert(m);
            Ok(self)
        } else {
            Err("Mass already exists in the System".to_string())
        }
    }

    /// Removes a `Mass` and all connecting `Spring`s from the `System` and
    /// returns the connecting `Spring`s.
    pub fn remove_mass(mut self, m_key: (usize, usize)) -> (Self, Vec<Spring>) {
        self.masses.remove(&m_key);
        let (connecting_springs, remaining_springs) = self.springs.drain(..).partition(|s| s.connects(m_key));
        self.springs = remaining_springs;
        (self.update_springs(), connecting_springs)
    }

    /// Insert a `Spring` to the `System`.
    ///
    /// # Errors
    ///
    /// - If both `Mass`es of the `Spring` are not in the `System`.
    pub fn with_spring(mut self, spring: Spring) -> Result<Self, String> {
        if self.masses.contains_key(&spring.a_key()) && self.masses.contains_key(&spring.b_key()) {
            self.springs.push(spring);
            Ok(self)
        } else {
            Err("Both Masses of the Spring must be in the System".to_string())
        }
    }

    /// Removes all springs that connect both `Mass`es and returns the removed
    /// `Spring`s.
    pub fn remove_springs_between(mut self, a_key: (usize, usize), b_key: (usize, usize)) -> (Self, Vec<Spring>) {
        let (connecting_springs, remaining_springs) = self
            .springs
            .drain(..)
            .partition(|s| s.connects(a_key) && s.connects(b_key));
        self.springs = remaining_springs;
        (self, connecting_springs)
    }

    /// Removes all springs whose spring constant is less than `k` and returns
    /// the removed `Spring`s.
    pub fn remove_loose_springs(mut self, k: f32) -> (Self, Vec<Spring>) {
        let (loose_springs, remaining_springs) = self.springs.drain(..).partition(|s| s.k() < k);
        self.springs = remaining_springs;
        (self, loose_springs)
    }

    /// Add springs between up to 10% of random pairs of `Mass`es.
    pub fn add_random_springs<I, D: Dataset<I>, M: Metric<I, T>>(mut self, data: &D, metric: &M, k: f32) -> Self {
        let mut pairs = Vec::new();
        let m_keys = self.masses.keys().copied().collect::<Vec<_>>();
        for i in 0..m_keys.len() {
            for j in (i + 1)..m_keys.len() {
                pairs.push((m_keys[i], m_keys[j]));
            }
        }
        let mut rng = StdRng::from_entropy();
        pairs.shuffle(&mut rng);
        pairs.truncate(pairs.len() / 100);

        self.springs.extend(pairs.into_iter().map(|(a_key, b_key)| {
            let l0 = data.one_to_one(a_key.1, b_key.1, metric);
            let l = self.masses[&a_key].current_distance_to(&self.masses[&b_key]);
            Spring::new(a_key, b_key, k, l0, l)
        }));

        self
    }

    /// Returns a `SizedHeap` of the `Mass`es in the `System` ordered by the
    /// stress on the `Mass`es.
    ///
    /// The stress is the sum of the magnitudes of the forces on the `Mass`.
    #[must_use]
    pub fn masses_by_stress(&self) -> SizedHeap<(f32, &Mass<DIM, T, S>)> {
        self.masses.values().map(|m| (m.stress(), m)).collect()
    }

    /// Returns up to the `n` most stressed `Mass`es in the `System` whose
    /// source is not a leaf.
    #[must_use]
    pub fn most_stressed_non_leaves(&self, n: usize) -> Vec<&Mass<DIM, T, S>> {
        let n_masses = Ord::max(self.masses.len() / 10, n);
        self.masses_by_stress()
            .items()
            .map(|(_, m)| m)
            .filter(|m| !m.source().is_leaf())
            .take(n_masses)
            .collect()
    }

    /// Finds all springs connecting to the given `Mass` and returns them.
    #[must_use]
    pub fn springs_connecting_to(&self, m_key: (usize, usize)) -> Vec<&Spring> {
        self.springs.iter().filter(|s| s.connects(m_key)).collect()
    }

    /// Finds all neighboring `Mass`es of the given `Mass` and returns them with
    /// the spring constant of the springs connecting them.
    fn neighbors_of(&self, m_key: (usize, usize)) -> Vec<((usize, usize), f32)> {
        self.springs_connecting_to(m_key)
            .into_iter()
            .map(|s| {
                let other = if s.a_key() == m_key { s.b_key() } else { s.a_key() };
                (other, s.k())
            })
            .collect()
    }

    /// Adds two `Mass`es to the `System` using the triangle between the given
    /// `Mass` and its two child masses.
    ///
    /// The child masses will also inherit the springs connecting the given
    /// parent mass to its neighbors. The inherited springs will have their
    /// spring constant multiplied by the given factor `f`.
    ///
    /// After adding the new masses and springs to the system, all springs will
    /// be updated using the [`update_springs`](Self::update_springs) method.
    ///
    /// # Arguments
    ///
    /// - `m`: The `Mass` from which to create the triangle with its children.
    /// - `data`: The dataset.
    /// - `metric`: The metric to use for the triangle.
    /// - `seed`: The seed for the random number generator.
    /// - `k`: The spring constant of the springs connecting the masses.
    /// - `f`: The factor by which to multiply the spring constant of the
    ///   springs inherited from the parent mass.
    ///
    /// # Returns
    ///
    /// The two new `Mass`es added to the `System`.
    #[allow(clippy::similar_names)]
    pub fn add_child_triangle<I, D: Dataset<I>, M: Metric<I, T>>(
        mut self,
        m_key: (usize, usize),
        data: &D,
        metric: &M,
        seed: Option<u64>,
        k: f32,
        f: f32,
    ) -> (Self, Option<[(usize, usize); 2]>) {
        let child_keys = if let Some((a, b, ma, mb, ab)) = self
            .masses
            .get(&m_key)
            .and_then(|m| m.child_triangle(data, metric, seed))
        {
            // Add the new masses to the system.
            let (a_center, b_center) = (a.arg_center(), b.arg_center());
            let [a_key, b_key] = [a.hash_key(), b.hash_key()];
            self = self
                .with_mass(a)
                .unwrap_or_else(|e| unreachable!("New mass is not in the system: {e}"))
                .with_mass(b)
                .unwrap_or_else(|e| unreachable!("New mass is not in the system: {e}"));

            // Create three new springs connecting the three masses.
            let ma_spring = Spring::new(a_key, m_key, k, ma, ma.as_f32());
            let mb_spring = Spring::new(m_key, b_key, k, mb, mb.as_f32());
            let ab_spring = Spring::new(a_key, b_key, k, ab, ab.as_f32());

            // The child masses will inherit the springs from the parent mass.
            self = self
                .neighbors_of(m_key)
                .into_iter()
                .flat_map(|(o_key, k)| {
                    let k = k * f;
                    let ao = data.one_to_one(a_center, o_key.1, metric);
                    let bo = data.one_to_one(b_center, o_key.1, metric);
                    [
                        Spring::new(a_key, o_key, k, ao, ao.as_f32()),
                        Spring::new(o_key, b_key, k, bo, bo.as_f32()),
                    ]
                })
                // Include the new springs for the triangle between the parent
                // and child masses.
                .chain([ma_spring, mb_spring, ab_spring])
                .fold(self, |system, s| {
                    system
                        .with_spring(s)
                        .unwrap_or_else(|e| unreachable!("All masses are part of the system: {e}"))
                });

            // Update the springs in the system.
            self = self.update_springs();

            Some([a_key, b_key])
        } else {
            None
        };

        (self, child_keys)
    }
}

impl<'a, const DIM: usize, T: Number, S: ParCluster<T>> System<'a, DIM, T, S> {
    /// Parallel version of [`System::evolve_to_leaves`](Self::evolve_to_leaves).
    #[allow(clippy::too_many_arguments)]
    pub fn par_evolve_to_leaves<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>, P: AsRef<std::path::Path>>(
        mut self,
        data: &D,
        metric: &M,
        seed: Option<u64>,
        k: f32,
        f: f32,
        min_k: Option<f32>,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
        out_dir: &P,
    ) -> Self {
        let f = f.clamp(0.0, 1.0);

        let mut m_keys = self
            .most_stressed_non_leaves(100)
            .into_iter()
            .map(Mass::hash_key)
            .collect::<Vec<_>>();
        if let Some(min_k) = min_k {
            while !m_keys.is_empty() {
                self = m_keys.iter().fold(self, |system, m_key| {
                    system.add_child_triangle(*m_key, data, metric, seed, k, f).0
                });
                self = self
                    .remove_loose_springs(min_k)
                    .0
                    .par_add_random_springs(data, metric, k)
                    .par_evolve_to_stability(dt, patience, target, max_steps);
                self = m_keys
                    .iter()
                    .fold(self, |system, &m_key| system.par_remove_mass(m_key).0);
                self = self.clear_energies();
                m_keys = self
                    .most_stressed_non_leaves(100)
                    .into_iter()
                    .map(Mass::hash_key)
                    .collect::<Vec<_>>();

                let path = out_dir
                    .as_ref()
                    .join(format!("{}-step-{}.npy", self.name, self.masses.len()));
                self.get_reduced_embedding()
                    .write_npy(&path)
                    .unwrap_or_else(|e| unreachable!("Error writing embeddings: {e}"));
            }
        } else {
            while !m_keys.is_empty() {
                self = m_keys.iter().fold(self, |system, m_key| {
                    system.add_child_triangle(*m_key, data, metric, seed, k, f).0
                });
                self = self
                    .par_add_random_springs(data, metric, k)
                    .par_evolve_to_stability(dt, patience, target, max_steps);
                self = m_keys
                    .iter()
                    .fold(self, |system, &m_key| system.par_remove_mass(m_key).0);
                self = self.clear_energies();
                m_keys = self
                    .most_stressed_non_leaves(100)
                    .into_iter()
                    .map(Mass::hash_key)
                    .collect::<Vec<_>>();

                let path = out_dir
                    .as_ref()
                    .join(format!("{}-step-{}.npy", self.name, self.masses.len()));
                self.get_reduced_embedding()
                    .write_npy(&path)
                    .unwrap_or_else(|e| unreachable!("Error writing embeddings: {e}"));
            }
        }

        self.par_evolve_to_stability(dt, patience * 10, target, None)
    }

    /// Parallel version of [`System::remove_mass`](Self::remove_mass).
    pub fn par_remove_mass(mut self, m_key: (usize, usize)) -> (Self, Vec<Spring>) {
        self.masses.remove(&m_key);
        let (connecting_springs, remaining_springs) = self.springs.drain(..).partition(|s| s.connects(m_key));
        self.springs = remaining_springs;
        (self.par_update_springs(), connecting_springs)
    }

    /// Parallel version of [`System::add_random_springs`](Self::add_random_springs).
    pub fn par_add_random_springs<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(
        mut self,
        data: &D,
        metric: &M,
        k: f32,
    ) -> Self {
        let mut pairs = Vec::new();
        let m_keys = self.masses.keys().copied().collect::<Vec<_>>();
        for i in 0..m_keys.len() {
            for j in (i + 1)..m_keys.len() {
                pairs.push((m_keys[i], m_keys[j]));
            }
        }
        let mut rng = StdRng::from_entropy();
        pairs.shuffle(&mut rng);
        pairs.truncate(pairs.len() / 100);

        self.springs.extend(
            pairs
                .into_par_iter()
                .map(|(a_key, b_key)| {
                    let l0 = data.par_one_to_one(a_key.1, b_key.1, metric);
                    let l = self.masses[&a_key].current_distance_to(&self.masses[&b_key]);
                    Spring::new(a_key, b_key, k, l0, l)
                })
                .collect::<Vec<_>>(),
        );

        self
    }

    /// Parallel version of [`System::from_graph`](Self::from_graph).
    pub fn par_from_graph(g: &'a Graph<T, S>, beta: f32, k: f32, name: &str) -> Self {
        let masses = g
            .iter_clusters()
            .map(Adapted::source)
            .map(Mass::new)
            .map(|m| (m.hash_key(), m))
            .collect::<HashMap<_, _>>();
        let springs = g
            .par_iter_edges()
            .map(|(a, b, l0)| {
                let (a_key, b_key) = (a.unique_id(), b.unique_id());
                let l = masses[&a_key].current_distance_to(&masses[&b_key]);
                Spring::new(a_key, b_key, k, l0, l)
            })
            .collect();
        Self {
            masses,
            springs,
            beta,
            energies: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Parallel version of [`System::init_random`](Self::init_random).
    pub fn par_init_random(mut self, side_length: f32, seed: Option<u64>) -> Self {
        let (min_, max_) = (-side_length / 2.0, side_length / 2.0);
        self.masses.par_iter_mut().for_each(|(&(c, _), m)| {
            let mut rng = seed.map_or_else(StdRng::from_entropy, |s| StdRng::seed_from_u64(s + c.as_u64()));
            let mut position = [0.0; DIM];
            for p in &mut position {
                *p = rng.gen_range(min_..max_);
            }
            m.set_position(position);
        });

        self
    }

    /// Parallel version of [`System::update_springs`](Self::update_springs).
    pub fn par_update_springs(mut self) -> Self {
        self.springs.par_iter_mut().for_each(|s| {
            let l = self.masses[&s.a_key()].current_distance_to(&self.masses[&s.b_key()]);
            s.update_length(l);
        });
        self
    }

    /// Parallel version of [`System::update_step`](Self::update_step).
    pub fn par_update_step(mut self, dt: f32) -> Self {
        let forces = self
            .springs
            .par_iter()
            .map(|s| {
                let f_mag = s.f_mag();
                let mut fv = self.masses[&s.a_key()].unit_vector_to(&self.masses[&s.b_key()]);
                for f_i in &mut fv {
                    *f_i *= f_mag;
                }
                (s.a_key(), s.b_key(), fv)
            })
            .collect::<Vec<_>>();

        for (a_key, b_key, f) in forces {
            if let Some(m) = self.masses.get_mut(&a_key) {
                m.add_force(f);
            }
            if let Some(m) = self.masses.get_mut(&b_key) {
                m.sub_force(f);
            }
        }

        self.masses
            .par_iter_mut()
            .for_each(|(_, m)| m.apply_force(dt, self.beta));

        self.energies.push(self.current_energies());

        self.par_update_springs()
    }

    /// Parallel version of [`System::evolve_to_stability`](Self::evolve_to_stability).
    pub fn par_evolve_to_stability(
        mut self,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
    ) -> Self {
        let target = target.unwrap_or(0.995);
        let max_steps = max_steps.unwrap_or(usize::MAX);

        self.energies.push(self.current_energies());

        let mut i = 0;
        let mut stability = self.stability(patience);
        while stability > target && i < max_steps {
            self = self.par_update_step(dt);
            i += 1;
            stability = self.stability(patience);
        }

        ftlog::debug!(
            "Reached stability: {stability:.10} after {i} steps with {} objects and {} springs",
            self.masses.len(),
            self.springs.len()
        );

        self
    }

    /// Parallel version of [`System::evolve`](Self::evolve).
    pub fn par_evolve(mut self, dt: f32, steps: usize) -> Self {
        self.energies.push(self.current_energies());

        for i in 0..steps {
            ftlog::debug!("Step {}/{steps}", i + 1);
            self = self.par_update_step(dt);
        }

        self
    }

    /// Parallel version of [`System::evolve_to_stability_with_saves`](Self::evolve_to_stability_with_saves).
    ///
    /// # Errors
    ///
    /// See [`System::evolve_to_stability_with_saves`](Self::evolve_to_stability_with_saves).
    #[cfg(feature = "disk-io")]
    #[allow(clippy::too_many_arguments)]
    pub fn par_evolve_to_stability_with_saves<P: AsRef<std::path::Path>>(
        mut self,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
        save_every: usize,
        dir: P,
        name: &str,
    ) -> Result<Self, String> {
        let target = target.unwrap_or(0.995);
        let max_steps = max_steps.unwrap_or(usize::MAX);
        self.energies.push(self.current_energies());

        let mut i = 0;
        let mut stability = self.stability(patience);
        while stability.is_nan() || (stability < target && i < max_steps) {
            if i % save_every == 0 {
                ftlog::debug!("{name}: Saving step {}", i + 1);
                let path = dir.as_ref().join(format!("{}.npy", i + 1));
                self.get_reduced_embedding().write_npy(&path)?;
            }

            ftlog::debug!(
                "Reached stability: {stability:.6} after {i} steps with {} objects and {} springs",
                self.masses.len(),
                self.springs.len()
            );
            self = self.par_update_step(dt);
            i += 1;
            stability = self.stability(patience);
        }

        ftlog::debug!(
            "Reached stability: {stability:.6} after {i} steps with {} objects and {} springs",
            self.masses.len(),
            self.springs.len()
        );

        Ok(self)
    }

    /// Parallel version of [`System::evolve_with_saves`](Self::evolve_with_saves).
    ///
    /// # Errors
    ///
    /// See [`System::evolve_with_saves`](Self::evolve_with_saves).
    #[cfg(feature = "disk-io")]
    pub fn par_evolve_with_saves<P: AsRef<std::path::Path>>(
        mut self,
        dt: f32,
        steps: usize,
        save_every: usize,
        dir: P,
        name: &str,
    ) -> Result<Self, String> {
        self.energies.push(self.current_energies());

        for i in 0..steps {
            if i % save_every == 0 {
                ftlog::debug!("{name}: Saving step {}/{steps}", i + 1);
                let path = dir.as_ref().join(format!("{}.npy", i + 1));
                self.get_reduced_embedding().write_npy(&path)?;
            }
            self = self.par_update_step(dt);
        }

        Ok(self)
    }
}

#[cfg(feature = "disk-io")]
impl<'a, const DIM: usize, T: Number + 'a, S: Cluster<T>> System<'a, DIM, T, S> {
    /// Encodes the `System` to a binary representation using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If there is an error encoding any `Mass` or `Spring`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();

        // Encode the masses.
        let masses = self.masses.values().map(super::mass::MassIO::from).collect::<Vec<_>>();
        let masses_bytes = bitcode::encode(&masses).map_err(|e| format!("Error encoding Masses: {e}"))?;
        bytes.extend_from_slice(&masses_bytes.len().to_le_bytes());
        bytes.extend_from_slice(&masses_bytes);

        // Encode the springs.
        let springs_bytes = bitcode::encode(&self.springs).map_err(|e| format!("Error encoding Springs: {e}"))?;
        bytes.extend_from_slice(&springs_bytes.len().to_le_bytes());
        bytes.extend_from_slice(&springs_bytes);

        // Encode the beta value.
        bytes.extend_from_slice(&self.beta.to_le_bytes());

        // Encode the energies.
        let energies = self.energies.iter().map(|v| v.to_vec()).collect::<Vec<_>>();
        let energies_bytes = bitcode::encode(&energies).map_err(|e| format!("Error encoding Energies: {e}"))?;
        bytes.extend_from_slice(&energies_bytes.len().to_le_bytes());
        bytes.extend_from_slice(&energies_bytes);

        // Encode the name.
        bytes.extend_from_slice(self.name.as_bytes());

        Ok(bytes)
    }

    /// Decodes a `System` from a binary representation using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If there is an error decoding any `Mass` or `Spring`.
    pub fn from_bytes(bytes: &[u8], root: &'a S) -> Result<Self, String> {
        let sources = root
            .subtree()
            .into_iter()
            .map(|c| (c.unique_id(), c))
            .collect::<HashMap<_, _>>();

        let mut offset = 0;

        // Decode the masses.
        let masses_bytes = crate::utils::read_encoding(bytes, &mut offset);
        let masses = bitcode::decode::<Vec<super::mass::MassIO>>(&masses_bytes)
            .map_err(|e| format!("Error decoding Masses: {e}"))?
            .iter()
            .map(|m| Mass::from_io(m, sources[&m.key]).map(|m| (m.hash_key(), m)))
            .collect::<Result<_, _>>()?;

        // Decode the springs.
        let springs_bytes = crate::utils::read_encoding(bytes, &mut offset);
        let springs = bitcode::decode(&springs_bytes).map_err(|e| format!("Error decoding Springs: {e}"))?;

        // Decode the beta value.
        let f32_size = core::mem::size_of::<f32>();
        let beta = <f32 as Number>::from_le_bytes(&bytes[offset..(offset + f32_size)]);
        offset += f32_size;

        // Decode the energies.
        let energies_bytes = crate::utils::read_encoding(bytes, &mut offset);
        let energies = bitcode::decode::<Vec<Vec<f32>>>(&energies_bytes)
            .map_err(|e| format!("Error decoding Energies: {e}"))?
            .into_iter()
            .map(|v| {
                let mut arr = [0.0; 3];
                arr.copy_from_slice(&v);
                arr
            })
            .collect();

        let name = std::str::from_utf8(&bytes[offset..])
            .map_err(|e| format!("Error decoding Name: {e}"))?
            .to_string();

        Ok(Self {
            masses,
            springs,
            beta,
            energies,
            name,
        })
    }

    /// Writes the `System` to a file at the given path.
    ///
    /// # Errors
    ///
    /// - See [`System::to_bytes`](Self::to_bytes).
    /// - If there is an error writing the bytes to the file.
    pub fn write_to<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, &bytes).map_err(|e| format!("Error writing System to file: {e}"))
    }

    /// Reads a `System` from a file at the given path.
    ///
    /// # Errors
    ///
    /// - If there is an error reading the bytes from the file.
    /// - See [`System::from_bytes`](Self::from_bytes).
    pub fn read_from<P: AsRef<std::path::Path>>(path: P, root: &'a S) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Error reading System from file: {e}"))?;
        Self::from_bytes(&bytes, root)
    }
}
