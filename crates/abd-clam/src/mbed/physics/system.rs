//! A mass-spring system for dimension reduction.

use std::collections::{hash_map::Entry, HashMap};

use distances::Number;
use rand::prelude::*;

use crate::{chaoda::Graph, core::adapters::Adapted, Cluster, Dataset, FlatVec, Metric, SizedHeap};

use super::{Mass, Spring};

/// A `HashMap` of `Mass`es from their hash keys.
pub type Masses<'a, const DIM: usize, T, S> = HashMap<(usize, usize), Mass<'a, DIM, T, S>>;

/// A mass-spring system for dimension reduction.
pub struct System<'a, const DIM: usize, T: Number, S: Cluster<T>> {
    /// The masses in the system.
    masses: Masses<'a, DIM, T, S>,
    /// The springs in the system.
    springs: Vec<Spring>,
    /// The damping factor of the system.
    beta: f32,
    /// The energy values of the system as it evolved over time.
    energies: Vec<[f32; 3]>,
}

/// Get the hash key of a `Vertex` for use in the `System`.
fn c_hash_key<T, C>(c: &C) -> (usize, usize)
where
    T: Number,
    C: Cluster<T>,
{
    (c.arg_center(), c.cardinality())
}

impl<'a, const DIM: usize, T: Number, S: Cluster<T>> System<'a, DIM, T, S> {
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
    #[must_use]
    pub fn from_graph(g: &'a Graph<T, S>, beta: f32, k: f32) -> Self {
        let masses = g
            .iter_clusters()
            .map(Adapted::source)
            .map(Mass::new)
            .map(|m| (m.hash_key(), m))
            .collect::<HashMap<_, _>>();
        let springs = g
            .iter_edges()
            .map(|(a, b, l0)| {
                let (a_key, b_key) = (c_hash_key(a), c_hash_key(b));
                let l = masses[&a_key].current_distance_to(&masses[&b_key]);
                Spring::new(a_key, b_key, k, l0, l)
            })
            .collect();
        Self {
            masses,
            springs,
            beta,
            energies: Vec::new(),
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
    pub const fn masses(&self) -> &Masses<DIM, T, S> {
        &self.masses
    }

    /// Returns the springs in the system.
    #[must_use]
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

    /// Updates the lengths and forces of the springs in the system.
    pub fn update_springs(&mut self) {
        self.springs.iter_mut().for_each(|s| {
            let l = self.masses[&s.a_key()].current_distance_to(&self.masses[&s.b_key()]);
            s.update_length(l);
        });
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
            .iter()
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

        self.masses.iter_mut().for_each(|(_, m)| m.apply_force(dt, self.beta));

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
        self.springs.iter().map(Spring::potential_energy).sum()
    }

    /// Get the total kinetic energy of the `System`.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.masses.values().map(Mass::kinetic_energy).sum()
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
    /// The stability is the mean of the `1 - (std-dev / mean_val)` of the
    /// kinetic and potential energies.
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

    /// Adds a `Mass` to the `System`.
    ///
    /// # Errors
    ///
    /// - If the `Mass` already exists in the `System`.
    pub fn add_mass(&mut self, m: Mass<'a, DIM, T, S>) -> Result<(), String> {
        if let Entry::Vacant(e) = self.masses.entry(m.hash_key()) {
            e.insert(m);
            Ok(())
        } else {
            Err("Mass already exists in the System".to_string())
        }
    }

    /// Removes a `Mass` and all connecting `Spring`s from the `System` and
    /// returns the connecting `Spring`s.
    pub fn remove_mass(&mut self, m: &Mass<DIM, T, S>) -> Vec<Spring> {
        let m_key = m.hash_key();
        self.masses.remove(&m_key);
        let (connecting_springs, remaining_springs) = self.springs.drain(..).partition(|s| s.connects(m_key));
        self.springs = remaining_springs;
        connecting_springs
    }

    /// Insert a `Spring` to the `System`.
    ///
    /// # Errors
    ///
    /// - If both `Mass`es of the `Spring` are not in the `System`.
    pub fn add_spring(&mut self, spring: Spring) -> Result<(), String> {
        if self.masses.contains_key(&spring.a_key()) && self.masses.contains_key(&spring.b_key()) {
            self.springs.push(spring);
            Ok(())
        } else {
            Err("Both Masses of the Spring must be in the System".to_string())
        }
    }

    /// Removes all springs that connect both `Mass`es and returns the removed
    /// `Spring`s.
    pub fn remove_springs_between(&mut self, a: &Mass<DIM, T, S>, b: &Mass<DIM, T, S>) -> Vec<Spring> {
        let (connecting_springs, remaining_springs) = self
            .springs
            .drain(..)
            .partition(|s| s.connects(a.hash_key()) && s.connects(b.hash_key()));
        self.springs = remaining_springs;
        connecting_springs
    }

    /// Removes all springs whose spring constant is less than `k` and returns
    /// the removed `Spring`s.
    pub fn remove_loose_springs(&mut self, k: f32) -> Vec<Spring> {
        let (loose_springs, remaining_springs) = self.springs.drain(..).partition(|s| s.k() < k);
        self.springs = remaining_springs;
        loose_springs
    }

    /// Returns a `SizedHeap` of the `Mass`es in the `System` ordered by the
    /// stress on the `Mass`es.
    ///
    /// The stress is the sum of the magnitudes of the forces on the `Mass`.
    #[must_use]
    pub fn masses_by_stress(&self) -> SizedHeap<(f32, &Mass<DIM, T, S>)> {
        self.masses.values().map(|m| (m.stress(), m)).collect()
    }

    /// Returns the most stressed `Mass` in the `System` whose source is not a
    /// leaf `Cluster`.
    #[must_use]
    pub fn most_stressed_non_leaf(&self) -> Option<&Mass<DIM, T, S>> {
        self.masses_by_stress()
            .items()
            .map(|(_, m)| m)
            .find(|m| !m.source().is_leaf())
    }

    /// Finds all springs connecting to the given `Mass` and returns them.
    #[must_use]
    pub fn springs_connecting_to(&self, m: &Mass<DIM, T, S>) -> Vec<&Spring> {
        self.springs.iter().filter(|s| s.connects(m.hash_key())).collect()
    }

    /// Finds all neighboring `Mass`es of the given `Mass` and returns them with
    /// the spring constant of the springs connecting them.
    fn neighbors_of(&self, m: &Mass<DIM, T, S>) -> Vec<((usize, usize), f32)> {
        self.springs_connecting_to(m)
            .into_iter()
            .map(|s| {
                let other = if s.a_key() == m.hash_key() {
                    s.b_key()
                } else {
                    s.a_key()
                };
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
        &'a mut self,
        m: &'a Mass<'a, DIM, T, S>,
        data: &D,
        metric: &M,
        seed: Option<u64>,
        k: f32,
        f: f32,
    ) -> Option<[&'a Mass<'a, DIM, T, S>; 2]> {
        if let Some((a, b, ma, mb, ab)) = m.child_triangle(data, metric, seed) {
            // Add the new masses to the system.
            let (a_center, b_center) = (a.arg_center(), b.arg_center());
            let (m_key, a_key, b_key) = (m.hash_key(), a.hash_key(), b.hash_key());
            self.add_mass(a)
                .unwrap_or_else(|e| unreachable!("New mass is not in the system: {e}"));
            self.add_mass(b)
                .unwrap_or_else(|e| unreachable!("New mass is not in the system: {e}"));

            // Create three new springs connecting the three masses.
            let ma_spring = Spring::new(a_key, m_key, k, ma, ma.as_f32());
            let mb_spring = Spring::new(m_key, b_key, k, mb, mb.as_f32());
            let ab_spring = Spring::new(a_key, b_key, k, ab, ab.as_f32());

            // The child masses will inherit the springs from the parent mass.
            self.neighbors_of(m)
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
                .for_each(|s| {
                    self.add_spring(s)
                        .unwrap_or_else(|e| unreachable!("All masses are part of the system: {e}"));
                });

            // Update the springs in the system.
            self.update_springs();

            Some([&self.masses[&a_key], &self.masses[&b_key]])
        } else {
            None
        }
    }
}
