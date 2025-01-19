//! A mass-spring system for dimension reduction.

use std::collections::HashMap;

use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    cakes::PermutedBall,
    chaoda::{Graph, Vertex},
    Cluster, FlatVec,
};

use super::{Mass, Spring};

/// A `HashMap` of `Mass`es from their hash keys.
pub type Masses<const DIM: usize> = HashMap<(usize, usize), Mass<DIM>>;

/// A mass-spring system for dimension reduction.
pub struct System<'a, const DIM: usize> {
    /// The masses in the system.
    masses: Masses<DIM>,
    /// The springs in the system.
    springs: Vec<Spring<'a, DIM>>,
    /// The damping factor of the system.
    beta: f32,
    /// The energy values of the system as it evolved over time.
    energies: Vec<[f32; 3]>,
}

/// Get the hash key of a `Vertex` for use in the `System`.
fn c_hash_key<T, C>(c: &Vertex<T, PermutedBall<T, C>>) -> (usize, usize)
where
    T: Number,
    C: Cluster<T>,
{
    (c.source.offset(), c.cardinality())
}

impl<'a, const DIM: usize> System<'a, DIM> {
    /// Creates a new `System` of `Mass`es from a `Graph`.
    ///
    /// The user will still need to set the `Springs` of the `System`, using the
    /// `reset_springs` method.
    ///
    /// # Arguments
    ///
    /// - `g`: The `Graph` to create the `System` from.
    /// - `beta`: The damping factor of the `System`.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the points in the original embedding space.
    /// - `U`: The type of the distance values.
    /// - `D`: The dataset.
    /// - `C`: The type of the `Cluster`s in the `Graph`.
    #[must_use]
    pub fn from_graph<T, C>(g: &Graph<T, PermutedBall<T, C>>, beta: f32) -> Self
    where
        T: Number,
        C: Cluster<T>,
    {
        let masses = g
            .iter_clusters()
            .map(Mass::<DIM>::from_vertex)
            .map(|m| (m.hash_key(), m))
            .collect();
        Self {
            masses,
            springs: Vec::new(),
            beta,
            energies: Vec::new(),
        }
    }

    /// Resets the `System`'s `Springs` to match the `Graph`.
    pub fn reset_springs<T, C>(&'a mut self, g: &Graph<T, PermutedBall<T, C>>, k: f32)
    where
        T: Number,
        C: Cluster<T>,
    {
        self.springs = g
            .iter_edges()
            .map(|(a, b, l0)| {
                let a = &self.masses[&c_hash_key(a)];
                let b = &self.masses[&c_hash_key(b)];
                Spring::new(a, b, k, l0)
            })
            .collect();
    }

    /// Sets random positions for all masses.
    ///
    /// The positions will be inside a cube with side length `side_length` centered at the origin.
    ///
    /// # Arguments
    ///
    /// - `side_length`: The side length of the cube.
    /// - `seed`: The seed for the random number generator.
    #[must_use]
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
    pub const fn masses(&self) -> &Masses<DIM> {
        &self.masses
    }

    /// Returns the springs in the system.
    #[must_use]
    pub fn springs(&self) -> &[Spring<'a, DIM>] {
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

    /// Updates the lengths and forces of the springs in the system.
    pub fn update_springs(&mut self) {
        self.springs.par_iter_mut().for_each(Spring::update_length);
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
    pub fn update_step(&mut self, dt: f32) {
        let forces = self
            .springs
            .par_iter()
            .map(|s| {
                let f_mag = s.f_mag();
                let mut fv = s.a.unit_vector_to(s.b);
                for f_i in &mut fv {
                    *f_i *= f_mag;
                }
                (s.a, s.b, fv)
            })
            .collect::<Vec<_>>();

        for (a, b, f) in forces {
            if let Some(m) = self.masses.get_mut(&a.hash_key()) {
                m.add_force(f);
            }
            if let Some(m) = self.masses.get_mut(&b.hash_key()) {
                m.sub_force(f);
            }
        }

        self.masses
            .par_iter_mut()
            .for_each(|(_, m)| m.apply_force(dt, self.beta));

        self.energies.push(self.current_energies());

        self.update_springs();
    }

    /// Simulate the `System` until it reaches a stable state.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `patience`: The number of time-steps to consider for stability.
    pub fn evolve_to_stability(&mut self, dt: f32, patience: usize, target: Option<f32>, max_steps: Option<usize>) {
        let target = target.unwrap_or(0.995);
        let max_steps = max_steps.unwrap_or(usize::MAX);

        self.energies.push(self.current_energies());

        let mut i = 0;
        let mut stability = self.stability(patience);
        while stability < target && i < max_steps {
            ftlog::debug!("Step {i}, Stability: {stability:.6}");
            self.update_step(dt);
            i += 1;
            stability = self.stability(patience);
        }
    }

    /// Simulate the `System` for a given number of time-steps.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `steps`: The number of time-steps to simulate.
    pub fn evolve(&mut self, dt: f32, steps: usize) {
        self.energies.push(self.current_energies());

        for i in 0..steps {
            ftlog::debug!("Step {}/{steps}", i + 1);
            self.update_step(dt);
        }
    }

    /// Get the dimension-reduced embedding from this `System`.
    ///
    /// This will return a `FlatVec` with the following properties:
    ///
    /// - The instances are the positions of the `Mass`es.
    /// - The metadata are the indices of the centers of the `Cluster`s
    ///   represented by the `Mass`es.
    /// - The distance function is the Euclidean distance.
    #[must_use]
    pub fn get_reduced_embedding(&self) -> FlatVec<Vec<f32>, usize> {
        let masses = {
            let mut masses = self.masses.iter().collect::<Vec<_>>();
            masses.sort_by(|&(a, _), &(b, _)| a.cmp(b));
            masses
        };

        let positions = masses
            .iter()
            .flat_map(|&(&(_, c), m)| (0..c).map(move |_| m.position().to_vec()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        // let distance_fn = |x: &Vec<f32>, y: &Vec<f32>| distances::simd::euclidean_f32(x, y);
        // let metric = Metric::new(distance_fn, false);

        FlatVec::new(positions).unwrap_or_else(|e| unreachable!("Error creating FlatVec: {e}"))
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
        &mut self,
        dt: f32,
        patience: usize,
        target: Option<f32>,
        max_steps: Option<usize>,
        save_every: usize,
        dir: P,
        name: &str,
    ) -> Result<(), String> {
        let target = target.unwrap_or(0.995);
        let max_steps = max_steps.unwrap_or(usize::MAX);
        self.energies.push(self.current_energies());

        let mut i = 0;
        let mut stability = self.stability(patience);
        while stability < target && i < max_steps {
            if i % save_every == 0 {
                ftlog::debug!("{name}: Saving step {}", i + 1);
                let path = dir.as_ref().join(format!("{}.npy", i + 1));
                self.get_reduced_embedding().write_npy(&path)?;
            }

            ftlog::debug!("Step {i}, Stability: {stability:.6}");
            self.update_step(dt);
            i += 1;
            stability = self.stability(patience);
        }

        Ok(())
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
        &mut self,
        dt: f32,
        steps: usize,
        save_every: usize,
        dir: P,
        name: &str,
    ) -> Result<(), String> {
        self.energies.push(self.current_energies());

        for i in 0..steps {
            if i % save_every == 0 {
                ftlog::debug!("{name}: Saving step {}/{steps}", i + 1);
                let path = dir.as_ref().join(format!("{}.npy", i + 1));
                self.get_reduced_embedding().write_npy(&path)?;
            }
            self.update_step(dt);
        }

        Ok(())
    }

    /// Get the total potential energy of the `System`.
    #[must_use]
    pub fn potential_energy(&self) -> f32 {
        self.springs.par_iter().map(Spring::potential_energy).sum()
    }

    /// Get the total kinetic energy of the `System`.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.masses.par_iter().map(|(_, m)| m.kinetic_energy()).sum()
    }

    /// Get the total energy of the `System`.
    #[must_use]
    pub fn total_energy(&self) -> f32 {
        self.potential_energy() + self.kinetic_energy()
    }

    /// Calculates the energy of the system.
    #[must_use]
    pub fn current_energies(&self) -> [f32; 3] {
        let potential_energy = self.potential_energy();
        let kinetic_energy = self.kinetic_energy();
        let total_energy = potential_energy + kinetic_energy;
        [potential_energy, kinetic_energy, total_energy]
    }

    /// Get the stability of the `System` over the last `n` time-steps.
    ///
    /// The stability is calculated as the mean of the `1 - (std-dev / mean_val)`
    /// of the kinetic and potential energies.
    ///
    /// # Arguments
    ///
    /// - `n`: The number of time-steps to consider.
    ///
    /// # Returns
    ///
    /// The stability of the `System` in a [0, 1] range, with 1 being stable.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn stability(&'a self, n: usize) -> f32 {
        if self.energies.len() < n {
            0.0
        } else {
            let last_n = &self.energies[(self.energies.len() - n)..];

            let (last_ke, last_pe) = last_n
                .iter()
                .map(|&[ke, pe, _]| (ke, pe))
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let stability_ke = {
                let mean = crate::utils::mean::<_, f32>(&last_ke);
                let variance = crate::utils::variance(&last_ke, mean);
                1.0 - (variance.sqrt() / mean)
            };

            let stability_pe = {
                let mean = crate::utils::mean::<_, f32>(&last_pe);
                let variance = crate::utils::variance(&last_pe, mean);
                1.0 - (variance.sqrt() / mean)
            };

            (stability_ke + stability_pe) / 2.0
        }
    }
}
