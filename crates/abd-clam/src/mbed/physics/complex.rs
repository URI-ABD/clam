//! A `Complex` is a set of `Mass`es and `Spring`s that are used to construct a
//! lower-dimensional embedding of the original dataset.

use distances::{number::Float, Number};
use rand::Rng;
use slotmap::HopSlotMap;

use crate::{Cluster, Dataset, FlatVec, Metric};

use super::{mass::Mass, spring::Spring, Vector};

// Define a key type for the HopSlotMap.
slotmap::new_key_type! { pub struct MassKey; }

/// Type alias for the `HopSlotMap` used to store masses.
pub type MassMap<'a, T, C, F, const DIM: usize> = HopSlotMap<MassKey, Mass<'a, T, C, F, DIM>>;

/// A type alias for a checkpoint in the simulation.
type Checkpoint<F, const DIM: usize> = (F, F, usize, FlatVec<[F; DIM], usize>);

/// Represents a collection of masses and springs.
pub struct Complex<'a, T, C, F, const DIM: usize>
where
    T: Number,
    C: Cluster<T>,
    F: Float,
{
    /// A map of masses, keyed by `MassKey`.
    masses: MassMap<'a, T, C, F, DIM>,
    /// A vector of springs connecting the masses.
    springs: Vec<Spring<F>>,
    /// A vector storing the kinetic and potential energies from previous time steps.
    energy_history: Vec<(F, F)>,
    /// The drag coefficient for the system.
    drag_coefficient: F,
    /// The scale factor for the rest length of the springs.
    scale: F,
    /// The spring constant to use for new springs.
    spring_constant: F,
    /// The loosening factor for springs, with a value between zero and one.
    loosening_factor: F,
}

impl<'a, T, C, F, const DIM: usize> Complex<'a, T, C, F, DIM>
where
    T: Number,
    C: Cluster<T>,
    F: Float,
{
    /// Creates a new `Complex` with a root `Cluster`.
    ///
    /// The first mass is placed at the origin with zero velocity.
    ///
    /// # Arguments
    ///
    /// - `root_cluster`: A reference to the root `Cluster`.
    /// - `drag_coefficient`: The drag coefficient for the system.
    /// - `scale`: The complex will try to keep masses inside a hyper-cube of
    ///   side length equals to twice this `scale`.
    /// - `spring_constant`: The spring constant to use for new springs.
    /// - `loosening_factor`: The loosening factor for inheriting springs, with a value between zero and one.
    pub fn new(root: &'a C, drag_coefficient: F, scale: F, spring_constant: F, loosening_factor: F) -> Self {
        let mut masses = MassMap::with_key();
        masses.insert(Mass::new(
            root,
            Vector::zero(), // Position at the origin.
            Vector::zero(), // Zero velocity.
        ));
        let scale = scale / F::from(root.radius());

        let mut complex = Self {
            masses,
            springs: Vec::new(),
            energy_history: Vec::new(),
            drag_coefficient,
            scale,
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
    pub fn simulate_to_leaves<R: Rng, I, D: Dataset<I>, M: Metric<I, T>>(
        &mut self,
        rng: &mut R,
        data: &D,
        metric: &M,
        max_steps: usize,
        tolerance: F,
        dt: F,
        n: usize,
        f: F,
    ) -> Vec<Checkpoint<F, DIM>> {
        let mut checkpoints = Vec::new();

        while self.masses.values().any(|m| !m.is_leaf()) {
            // Perform a major update.
            self.major_update(rng, dt, f, data, metric);

            // Prune springs that connect masses whose clusters are too far apart.
            self.prune_springs();

            // Allow the system to relax to equilibrium.
            let (ke, pe, steps) = self.relax_to_equilibrium(max_steps, tolerance, dt, n);

            // Record a checkpoint.
            let positions = self.extract_embedding();
            checkpoints.push((ke, pe, steps, positions));
        }

        checkpoints
    }

    /// Remove springs that connect masses whose clusters are too far apart.
    ///
    /// Two masses are considered too far apart if the distance between their
    /// clusters is greater than the sum of the radii of the clusters times the
    /// square root of 2. This check ensures that the neither the two clusters
    /// nor any of their descendants will have overlapping volumes in the original
    /// dataset.
    pub fn prune_springs(&mut self) {
        self.springs.retain(|s| {
            let [a, b] = s.mass_keys();
            let [a, b] = [&self.masses[a], &self.masses[b]];
            let threshold = a.radius() + b.radius() * F::SQRT_2;
            s.rest_length() <= threshold
        });
    }

    /// Records the kinetic and potential energies at the current time step.
    fn record_energy(&mut self) -> (F, F) {
        let kinetic_energy = self.total_kinetic_energy();
        let potential_energy = self.total_potential_energy();
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
    fn add_spring<I, D: Dataset<I>, M: Metric<I, T>>(
        &mut self,
        a_key: MassKey,
        b_key: MassKey,
        data: &D,
        metric: &M,
    ) -> &Spring<F> {
        let s = Spring::new(
            [a_key, b_key],
            &self.masses,
            self.spring_constant,
            data,
            metric,
            self.scale,
        );
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

    /// Calculates the total potential energy stored in all springs in the `Complex`.
    ///
    /// # Returns
    ///
    /// The total potential energy as a floating-point value.
    fn total_potential_energy(&self) -> F {
        self.springs.iter().map(Spring::potential_energy).sum()
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
            (F::ZERO, F::ZERO)
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
            .map_or((F::ZERO, F::ZERO), |&(ke, pe)| (ke, pe))
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
    pub fn major_update<R: Rng, I, D: Dataset<I>, M: Metric<I, T>>(
        &mut self,
        rng: &mut R,
        dt: F,
        f: F,
        data: &D,
        metric: &M,
    ) -> (F, F) {
        // Sort the non-leaf masses by the total force exerted on them.
        let mut masses = self
            .masses
            .iter()
            .filter_map(|(k, m)| {
                if m.is_leaf() {
                    None
                } else {
                    Some((k, m.total_force_magnitudes()))
                }
            })
            .collect::<Vec<_>>();
        masses.sort_by(|(_, a), (_, b)| b.total_cmp(a));

        // Remove the most stressed masses from the system.
        let n = (f * F::from(masses.len())).ceil().as_usize();
        let stressed_masses = masses
            .into_iter()
            .take(n)
            .map(|(k, _)| {
                (
                    k,
                    self.masses
                        .remove(k)
                        .unwrap_or_else(|| unreachable!("We know that the mass exists.")),
                )
            })
            .collect::<Vec<_>>();

        // Explode the most stressed masses.
        let children = stressed_masses.into_iter().map(|(k, m)| {
            let [a, b] = m.explode(rng, self.drag_coefficient, dt);
            (k, a, b)
        });

        // Insert the children back into the system.
        #[allow(clippy::needless_collect)]
        let triplet_keys = children
            .map(|(m, a, b)| (m, self.masses.insert(a), self.masses.insert(b)))
            .collect::<Vec<_>>();

        // Transfer springs from the exploded masses to their children and add springs between the siblings.
        for (m, a, b) in triplet_keys {
            let m_springs;
            (m_springs, self.springs) = self.springs.drain(..).partition(|s| s.mass_keys().contains(&m));
            let new_springs = m_springs
                .into_iter()
                .flat_map(|s| s.inherit([m, a, b], &self.masses, data, metric, self.scale, self.loosening_factor));
            self.springs.extend(new_springs);
            self.add_spring(a, b, data, metric);
        }

        // Record the energy history of the system.
        self.record_energy()
    }

    /// Extracts the positions of the items in the `Complex`.
    pub fn extract_embedding(&self) -> FlatVec<[F; DIM], usize> {
        let mut positions = self
            .masses
            .values()
            .flat_map(Mass::itemized_positions)
            .collect::<Vec<_>>();
        // sort by index
        positions.sort_by_key(|(i, _)| *i);
        let positions = positions.into_iter().map(|(_, pos)| pos.into()).collect();
        FlatVec::new(positions).unwrap_or_else(|_| unreachable!("We know that the system contains at least one mass."))
    }
}
