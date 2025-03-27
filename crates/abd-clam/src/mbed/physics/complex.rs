//! A `Complex` is a set of `Mass`es and `Spring`s that are used to construct a
//! lower-dimensional embedding of the original dataset.

use distances::{number::Float, Number};
use slotmap::HopSlotMap;

use crate::{Cluster, Dataset, FlatVec, Metric};

use super::{mass::Mass, spring::Spring, Vector};

// Define a key type for the HopSlotMap.
slotmap::new_key_type! { pub struct MassKey; }

/// Type alias for the `HopSlotMap` used to store masses.
pub type MassMap<'a, T, C, F, const DIM: usize> = HopSlotMap<MassKey, Mass<'a, T, C, F, DIM>>;

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
    pub fn new(root: &'a C, drag_coefficient: F, scale: F) -> Self {
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
        };

        // Record the initial energy history.
        complex.record_energy();

        complex
    }

    /// Records the kinetic and potential energies at the current time step.
    fn record_energy(&mut self) -> (F, F) {
        let kinetic_energy = self.total_kinetic_energy();
        let potential_energy = self.total_potential_energy();
        self.energy_history.push((kinetic_energy, potential_energy));
        (kinetic_energy, potential_energy)
    }

    /// Returns a reference to the masses.
    pub const fn masses(&self) -> &MassMap<'a, T, C, F, DIM> {
        &self.masses
    }

    /// Returns a reference to the springs.
    pub const fn springs(&self) -> &Vec<Spring<F>> {
        &self.springs
    }

    /// Creates a new spring connecting two masses.
    ///
    /// # Arguments
    ///
    /// - `a_key`: The key of the first mass.
    /// - `b_key`: The key of the second mass.
    /// - `spring_constant`: The spring constant.
    ///
    /// # Returns
    ///
    /// The new `Spring` connecting the two masses.
    pub fn new_spring<I, D: Dataset<I>, M: Metric<I, T>>(
        &self,
        a_key: MassKey,
        b_key: MassKey,
        spring_constant: F,
        data: &D,
        metric: &M,
    ) -> Spring<F> {
        Spring::new([a_key, b_key], &self.masses, spring_constant, data, metric, self.scale)
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

    /// Erases the energy history and keeps only the history for the previous `n` steps.
    ///
    /// # Arguments
    ///
    /// - `n`: The number of previous steps to retain.
    pub fn retain_energy_history(&mut self, n: usize) {
        if n == 0 {
            self.energy_history.clear();
            self.record_energy();
        } else if self.energy_history.len() > n {
            let start = self.energy_history.len() - n;
            self.energy_history = self.energy_history[start..].to_vec();
        }
    }

    /// Let the system evolve to equilibrium, or until the maximum number of steps is reached.
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
    pub fn evolve_to_equilibrium(&mut self, max_steps: usize, tolerance: F, dt: F, n: usize) -> (F, F, usize) {
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

    /// Updates the `Complex` for one time step.
    ///
    /// # Arguments
    ///
    /// - `dt`: The size of the time step.
    ///
    /// # Returns
    ///
    /// The kinetic and potential energies of the system after the update.
    fn minor_update(&mut self, dt: F) -> (F, F) {
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

    /// Extracts the positions of the items in the `Complex`.
    pub fn extract_embedding(&self) -> FlatVec<Vector<F, DIM>, usize> {
        let mut positions = self
            .masses
            .values()
            .flat_map(Mass::itemized_positions)
            .collect::<Vec<_>>();
        // sort by index
        positions.sort_by_key(|(i, _)| *i);
        let positions = positions.into_iter().map(|(_, pos)| pos).collect();
        FlatVec::new(positions).unwrap_or_else(|_| unreachable!("We know that the system contains at least one mass."))
    }
}
