//! A `Complex` is a set of `Mass`es and `Spring`s that are used to construct a
//! lower-dimensional embedding of the original dataset.

use rand::{distr::uniform::SampleUniform, Rng};
use rayon::prelude::*;
use slotmap::HopSlotMap;

use crate::{Cluster, Dataset, DistanceValue, FloatDistanceValue, ParCluster, ParDataset};

use super::{mass::Mass, spring::Spring, Vector};

// Define a key type for the HopSlotMap.
slotmap::new_key_type! { pub struct MassKey; }

/// Type alias for the `HopSlotMap` used to store masses.
pub type MassMap<'a, T, C, F, const DIM: usize> = HopSlotMap<MassKey, Mass<'a, T, C, F, DIM>>;

/// A type alias for a checkpoint in the simulation.
type Checkpoint<F, const DIM: usize> = Vec<[F; DIM]>;

/// Represents a collection of masses and springs.
pub struct Complex<'a, T, C, F, const DIM: usize>
where
    T: DistanceValue,
    C: Cluster<T>,
    F: FloatDistanceValue,
{
    /// A map of masses, keyed by `MassKey`.
    masses: MassMap<'a, T, C, F, DIM>,
    /// A vector of springs connecting the masses.
    springs: Vec<Spring<F>>,
    /// A vector storing the kinetic and potential energies from previous time steps.
    energy_history: Vec<(F, F)>,
    /// The drag coefficient for the system.
    drag_coefficient: F,
    /// The spring constant to use for new springs.
    spring_constant: F,
    /// The loosening factor for springs, with a value between zero and one.
    loosening_factor: F,
}

impl<'a, T, C, F, const DIM: usize> Complex<'a, T, C, F, DIM>
where
    T: DistanceValue,
    C: Cluster<T>,
    F: FloatDistanceValue + SampleUniform + core::fmt::Debug,
{
    /// Creates a new `Complex` with a root `Cluster`.
    ///
    /// The first mass is placed at the origin with zero velocity.
    ///
    /// # Arguments
    ///
    /// - `root_cluster`: A reference to the root `Cluster`.
    /// - `drag_coefficient`: The drag coefficient for the system.
    /// - `spring_constant`: The spring constant to use for new springs.
    /// - `loosening_factor`: The loosening factor for inheriting springs, with a value between zero and one.
    pub fn new(root: &'a C, drag_coefficient: F, spring_constant: F, loosening_factor: F) -> Self {
        let mut masses = MassMap::with_key();
        masses.insert(Mass::new(
            root,
            Vector::zero(), // Position at the origin.
            Vector::zero(), // Zero velocity.
        ));

        let mut complex = Self {
            masses,
            springs: Vec::new(),
            energy_history: Vec::new(),
            drag_coefficient,
            spring_constant,
            loosening_factor,
        };

        // Record the initial energy history.
        complex.record_energy();

        complex
    }

    /// Simulates the system until all masses represent leaf clusters.
    ///
    /// This method performs a series of major updates and relaxation steps to
    /// evolve the system. During each major update, the most stressed masses
    /// are exploded into their child clusters, and springs are recalculated
    /// accordingly. After each major update, the system is allowed to relax
    /// to equilibrium.
    ///
    /// # Arguments
    ///
    /// - `rng`: A mutable reference to a random number generator.
    /// - `data`: A reference to the dataset containing the items.
    /// - `metric`: The metric to use for distance calculation.
    /// - `max_steps`: The maximum number of steps to take during relaxation.
    /// - `tolerance`: The tolerance for the change in kinetic energy during relaxation.
    /// - `dt`: The size of the time step for the simulation.
    /// - `n`: The number of previous steps to consider for the tolerance.
    /// - `f`: The fraction of the most stressed masses to explode during each major update.
    ///
    /// # Returns
    ///
    /// A vector of checkpoints. Each checkpoint is collected after a major update
    /// and relaxation step, and contains the kinetic and potential energies of the
    /// system, the number of steps taken during relaxation, and the positions of
    /// the items in the system.
    #[allow(clippy::too_many_arguments)]
    pub fn simulate_to_leaves<R: Rng, I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &mut self,
        rng: &mut R,
        data: &D,
        metric: &M,
        max_steps: usize,
        tolerance: F,
        dt: F,
        n: usize,
    ) -> Vec<Checkpoint<F, DIM>> {
        let mut checkpoints = Vec::new();

        while self.masses.values().any(|m| !m.is_leaf()) {
            // Perform a major update.
            self.major_update(rng, dt, data, metric);

            // Allow the system to relax to equilibrium.
            let (ke, pe, steps) = self.relax_to_equilibrium(max_steps, tolerance, dt, n);

            // Prune springs that connect masses whose clusters are too far apart.
            self.prune_springs();

            // Stop all masses that are not connected to any springs.
            self.stop_isolated_masses();

            // Center the system at the origin.
            self.center_at_origin();

            // Record a checkpoint.
            checkpoints.push(self.extract_embedding());

            ftlog::info!(
                "Checkpoint {} recorded: steps: {steps}, ke: {:?}, pe: {:?} with {} masses and {} springs",
                checkpoints.len(),
                ke,
                pe,
                self.masses.len(),
                self.springs.len()
            );
        }

        while !self.springs.is_empty() {
            self.loosen_springs();
            let (ke, pe, steps) = self.relax_to_equilibrium(max_steps, tolerance, dt, n);
            self.center_at_origin();
            checkpoints.push(self.extract_embedding());
            ftlog::info!(
                "Checkpoint {} recorded: steps {steps}, ke: {:?}, pe: {:?} with {} masses and {} springs",
                checkpoints.len(),
                ke,
                pe,
                self.masses.len(),
                self.springs.len()
            );
            self.prune_springs();

            self.stop_isolated_masses();
        }

        checkpoints
    }

    /// For each mass in the system, add a spring to a randomly chosen different
    /// mass.
    ///
    /// # Arguments
    ///
    /// - `rng`: A mutable reference to a random number generator.
    /// - `data`: A reference to the dataset containing the items.
    /// - `metric`: The metric to use for distance calculation.
    pub fn add_random_springs<R: Rng, I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &mut self,
        rng: &mut R,
        data: &D,
        metric: &M,
    ) {
        let keys = self.masses.keys().collect::<Vec<_>>();
        let derangement = crate::utils::random_derangement(rng, keys.len());

        derangement
            .into_iter()
            .map(|i| &keys[i])
            .zip(keys.iter())
            .for_each(|(&a_key, &b_key)| {
                self.add_spring(a_key, b_key, data, metric);
            });
    }

    /// Remove springs that connect masses whose clusters are too far apart.
    ///
    /// Two masses are considered too far apart if the distance between their
    /// clusters is greater than the sum of the radii of the clusters times the
    /// square root of 2. This check ensures that the neither the two clusters
    /// nor any of their descendants will have overlapping volumes in the original
    /// dataset.
    pub fn prune_springs(&mut self) {
        let min_stiffness = self.spring_constant * self.loosening_factor.powi(7);
        self.springs.retain(|s| {
            // s.stiffness() >= min_stiffness
            s.stiffness() >= min_stiffness && {
                let [a, b] = s.mass_keys();
                let [a, b] = [&self.masses[a], &self.masses[b]];
                let threshold = a.radius() + b.radius();
                s.rest_length() <= threshold * (F::one() + F::one()).sqrt()
            }
        });
    }

    /// Records the kinetic and potential energies at the current time step.
    fn record_energy(&mut self) -> (F, F) {
        let kinetic_energy = self.mean_kinetic_energy();
        let potential_energy = self.mean_potential_energy();
        self.energy_history.push((kinetic_energy, potential_energy));
        (kinetic_energy, potential_energy)
    }

    /// Creates a new spring connecting two masses.
    ///
    /// # Arguments
    ///
    /// - `a_key`: The key of the first mass.
    /// - `b_key`: The key of the second mass.
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculation.
    ///
    /// # Returns
    ///
    /// The new `Spring` connecting the two masses.
    fn add_spring<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &mut self,
        a_key: MassKey,
        b_key: MassKey,
        data: &D,
        metric: &M,
    ) -> &Spring<F> {
        let s = Spring::new([a_key, b_key], &self.masses, self.spring_constant, data, metric);
        ftlog::debug!("Adding spring {s:?}");
        self.springs.push(s);
        self.springs
            .last()
            .unwrap_or_else(|| unreachable!("We know that the spring was just added."))
    }

    /// Calculates the total kinetic energy of all masses in the `Complex`.
    ///
    /// # Returns
    ///
    /// The total kinetic energy as a floating-point value.
    fn total_kinetic_energy(&self) -> F {
        self.masses.values().map(Mass::kinetic_energy).sum()
    }

    /// Calculates the mean kinetic energy of all masses in the `Complex`.
    fn mean_kinetic_energy(&self) -> F {
        self.total_kinetic_energy()
            / F::from(self.masses.len())
                .unwrap_or_else(|| unreachable!("We know that the system contains at least one mass."))
    }

    /// Calculates the total potential energy stored in all springs in the `Complex`.
    ///
    /// # Returns
    ///
    /// The total potential energy as a floating-point value.
    fn total_potential_energy(&self) -> F {
        self.springs.iter().map(Spring::potential_energy).sum()
    }

    /// Calculates the mean potential energy of all springs in the `Complex`.
    fn mean_potential_energy(&self) -> F {
        self.total_potential_energy()
            / F::from(self.springs.len())
                .unwrap_or_else(|| unreachable!("We know that the system contains at least one spring."))
    }

    /// Calculates the mean and standard deviation of the kinetic energy over the previous `n` steps.
    ///
    /// # Arguments
    ///
    /// - `n`: The number of previous steps to consider.
    ///
    /// # Returns
    ///
    /// A tuple containing the mean and standard deviation of the kinetic energy.
    fn kinetic_energy_stats(&self, n: usize) -> (F, F) {
        if n == 0 {
            (F::zero(), F::zero())
        } else {
            let len = self.energy_history.len();
            let start = len.saturating_sub(n);
            let kinetic_energies: Vec<F> = self.energy_history.iter().skip(start).map(|(ke, _)| *ke).collect();

            let mean_value = crate::utils::mean(&kinetic_energies);
            let std_dev = crate::utils::standard_deviation(&kinetic_energies);

            (mean_value, std_dev)
        }
    }

    /// Let the system relax to equilibrium, or until the maximum number of steps is reached.
    ///
    /// # Arguments
    ///
    /// - `max_steps`: The maximum number of steps to take.
    /// - `tolerance`: The tolerance for the change in kinetic energy.
    /// - `dt`: The size of the time step.
    /// - `n`: The number of previous steps to consider for the tolerance.
    ///
    /// # Returns
    ///
    /// The kinetic and potential energies of the system after the update, and the number of steps taken.
    pub fn relax_to_equilibrium(&mut self, max_steps: usize, tolerance: F, dt: F, n: usize) -> (F, F, usize) {
        let mut steps = 0;

        while steps < n || (steps < max_steps && !self.has_reached_equilibrium(tolerance, n)) {
            self.minor_update(dt);
            steps += 1;
        }

        let (ke, pe) = self.last_energy();
        (ke, pe, steps)
    }

    /// Returns the last recorded energy state of the system.
    fn last_energy(&self) -> (F, F) {
        self.energy_history
            .last()
            .map_or_else(|| (F::zero(), F::zero()), |&(ke, pe)| (ke, pe))
    }

    /// Returns the energy history of the system.
    pub const fn energy_history(&self) -> &Vec<(F, F)> {
        &self.energy_history
    }

    /// Checks whether the system has reached equilibrium.
    ///
    /// # Arguments
    ///
    /// - `tolerance`: The tolerance for the change in kinetic energy.
    /// - `n`: The number of previous steps to consider for the tolerance.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the system has reached equilibrium.
    pub fn has_reached_equilibrium(&self, tolerance: F, n: usize) -> bool {
        let (_, std_dev_ke) = self.kinetic_energy_stats(n);
        std_dev_ke < tolerance
    }

    /// Performs a minor update, moving the masses for one time step.
    ///
    /// # Arguments
    ///
    /// - `dt`: The size of the time step.
    ///
    /// # Returns
    ///
    /// The kinetic and potential energies of the system after the update.
    pub fn minor_update(&mut self, dt: F) -> (F, F) {
        // Step 1: Recalculate the forces exerted by springs.
        self.springs.iter_mut().for_each(|spring| {
            spring.recalculate(&self.masses);
            let force_vector = spring.force_vector(&self.masses);

            let [a_key, b_key] = spring.mass_keys();
            self.masses[a_key].add_force(&force_vector);
            self.masses[b_key].sub_force(&force_vector);
        });

        // Step 2: Move the masses for one time step.
        self.masses.values_mut().for_each(|mass| {
            mass.move_mass(self.drag_coefficient, dt);
        });

        // Step 3: Update the energy history of the system.
        self.record_energy()
    }

    /// Performs a major update, exploding the most stressed masses and recalculating the springs.
    ///
    /// # Arguments
    ///
    /// - `dt`: The size of the time step.
    /// - `f`: The fraction of the most stressed masses to explode.
    ///
    /// # Returns
    ///
    /// The kinetic and potential energies of the system after the update.
    pub fn major_update<R: Rng, I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &mut self,
        rng: &mut R,
        dt: F,
        data: &D,
        metric: &M,
    ) -> (F, F) {
        // Explode the non-leaf masses
        #[allow(clippy::needless_collect)]
        let children = self
            .masses
            .iter()
            .filter_map(|(k, m)| {
                if m.is_leaf() {
                    None
                } else {
                    let [a, b] = m.explode(rng, self.drag_coefficient, dt);
                    Some((k, a, b))
                }
            })
            .collect::<Vec<_>>();

        // Insert the children back into the system.
        #[allow(clippy::needless_collect)]
        let triplet_keys = children
            .into_iter()
            .map(|(m, a, b)| (m, self.masses.insert(a), self.masses.insert(b)))
            .collect::<Vec<_>>();

        // Transfer springs from the exploded masses to their children and add springs between the siblings.
        for &(m, a, b) in &triplet_keys {
            let new_springs = self
                .springs
                .iter()
                .filter_map(|s| {
                    if s.mass_keys().contains(&m) {
                        Some(s.inherit([m, a, b], &self.masses, data, metric, self.loosening_factor))
                    } else {
                        None
                    }
                })
                .flatten()
                .collect::<Vec<_>>();
            self.springs.extend(new_springs);
            self.add_spring(a, b, data, metric);
        }

        // Remove the stressed masses and all connected springs from the system.
        for (m, _, _) in triplet_keys {
            self.masses.remove(m);
            let n_springs = self.springs.len();
            self.springs.retain(|s| !s.mass_keys().contains(&m));
            let n_removed = n_springs - self.springs.len();
            ftlog::debug!("Removed {n_removed} springs connected to mass {m:?}");
        }

        // Record the energy history of the system.
        self.record_energy()
    }

    /// Extracts the positions of the items in the `Complex`.
    pub fn extract_embedding(&self) -> Vec<[F; DIM]> {
        let mut positions = self
            .masses
            .values()
            .flat_map(Mass::itemized_positions)
            .collect::<Vec<_>>();
        // sort by index
        positions.sort_by_key(|(i, _)| *i);
        positions.into_iter().map(|(_, pos)| pos.into()).collect()
    }

    /// Stop all masses that are not connected to any springs.
    pub fn stop_isolated_masses(&mut self) {
        self.masses.iter_mut().for_each(|(k, m)| {
            if !self.springs.iter().any(|s| s.mass_keys().contains(&k)) {
                m.set_velocity(Vector::zero());
            }
        });
    }

    /// Loosen all springs in the system.
    pub fn loosen_springs(&mut self) {
        self.springs.iter_mut().for_each(|s| s.loosen(self.loosening_factor));
    }

    /// Computes the location of the center of mass of the system.
    pub fn center_of_mass(&self) -> Vector<F, DIM> {
        let (pos, mass) = self
            .masses
            .values()
            .fold((Vector::zero(), F::zero()), |(pos, mass), m| {
                let m_pos = m.position();
                let m_mass = m.mass();
                (pos + m_pos * m_mass, mass + m_mass)
            });
        pos / mass
    }

    /// Moves all masses in the system so that the center of mass is at the origin.
    pub fn center_at_origin(&mut self) {
        let shift = -self.center_of_mass();
        self.masses.values_mut().for_each(|m| m.shift_position(shift));
    }
}

impl<T, C, F, const DIM: usize> Complex<'_, T, C, F, DIM>
where
    T: DistanceValue + Send + Sync,
    C: ParCluster<T>,
    F: FloatDistanceValue + SampleUniform + std::fmt::Debug + Send + Sync,
{
    /// Parallel version of the [`simulate_to_leaves`](Self::simulate_to_leaves) method.
    #[allow(clippy::too_many_arguments)]
    pub fn par_simulate_to_leaves<R: Rng, I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &mut self,
        rng: &mut R,
        data: &D,
        metric: &M,
        max_steps: usize,
        tolerance: F,
        dt: F,
        n: usize,
    ) -> Vec<Checkpoint<F, DIM>> {
        let mut checkpoints = Vec::new();

        while self.masses.values().any(|m| !m.is_leaf()) {
            // Perform a major update.
            self.major_update(rng, dt, data, metric);

            // Allow the system to relax to equilibrium.
            let (ke, pe, steps) = self.par_relax_to_equilibrium(max_steps, tolerance, dt, n);

            // Prune springs that connect masses whose clusters are too far apart.
            self.prune_springs();

            // Stop all masses that are not connected to any springs.
            self.stop_isolated_masses();

            // Center the system at the origin.
            self.center_at_origin();

            // Record a checkpoint.
            checkpoints.push(self.extract_embedding());

            ftlog::info!(
                "Checkpoint {} recorded: steps: {steps}, ke: {:?}, pe: {:?} with {} masses and {} springs",
                checkpoints.len(),
                ke,
                pe,
                self.masses.len(),
                self.springs.len()
            );
        }

        while !self.springs.is_empty() {
            self.par_loosen_springs();
            let (ke, pe, steps) = self.par_relax_to_equilibrium(max_steps, tolerance, dt, n);
            self.center_at_origin();
            checkpoints.push(self.extract_embedding());
            ftlog::info!(
                "Checkpoint {} recorded: steps {steps}, ke: {:?}, pe: {:?} with {} masses and {} springs",
                checkpoints.len(),
                ke,
                pe,
                self.masses.len(),
                self.springs.len()
            );
            self.prune_springs();

            self.stop_isolated_masses();
        }

        checkpoints
    }

    /// Parallel version of the [`relax_to_equilibrium`](Self::relax_to_equilibrium) method.
    pub fn par_relax_to_equilibrium(&mut self, max_steps: usize, tolerance: F, dt: F, n: usize) -> (F, F, usize) {
        let mut steps = 0;

        while steps < n || (steps < max_steps && !self.has_reached_equilibrium(tolerance, n)) {
            self.par_minor_update(dt);
            steps += 1;
        }

        let (ke, pe) = self.last_energy();
        (ke, pe, steps)
    }

    /// Parallel version of the [`minor_update`](Self::minor_update) method.
    pub fn par_minor_update(&mut self, dt: F) -> (F, F) {
        // Step 1: Recalculate the forces exerted by springs.
        self.springs.par_iter_mut().for_each(|spring| {
            spring.recalculate(&self.masses);
            let force_vector = spring.force_vector(&self.masses);

            let [a_key, b_key] = spring.mass_keys();
            self.masses[a_key].add_force(&force_vector);
            self.masses[b_key].sub_force(&force_vector);
        });

        // Step 2: Move the masses for one time step.
        self.masses.values_mut().for_each(|mass| {
            mass.move_mass(self.drag_coefficient, dt);
        });

        // Step 3: Update the energy history of the system.
        self.record_energy()
    }

    /// Parallel version of the [`loosen_springs`](Self::loosen_springs) method.
    pub fn par_loosen_springs(&mut self) {
        self.springs
            .par_iter_mut()
            .for_each(|s| s.loosen(self.loosening_factor));
    }
}
