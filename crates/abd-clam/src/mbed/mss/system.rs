//! The `Mass`-`Spring`-`System`.

use std::collections::HashMap;

use distances::Number;
use generational_arena::{Arena, Index};

use crate::{mbed::Vector, Cluster, Dataset, FlatVec, Metric};

use super::{Mass, Spring};

/// A `Mass`-`Spring`-`System`.
#[must_use]
pub struct System<'a, T: Number, C: Cluster<T>, const DIM: usize> {
    /// The drag coefficient.
    drag: f32,
    /// The spring constant of the primary springs.
    k: f32,
    /// The time-step of the simulation.
    dt: f32,
    /// The number of update-steps to wait before checking for equilibrium.
    patience: usize,
    /// The threshold of kinetic energy below which the system is considered to
    /// be in equilibrium.
    ke_threshold: f32,
    /// The maximum number of update-steps to run between major updates.
    max_steps: usize,
    /// The `Mass`es in the system, in a generational arena.
    masses: Arena<Mass<'a, T, C, DIM>>,
    /// The `Spring`s in the system which connect at least one non-leaf `Mass`.
    springs: Vec<Spring<DIM>>,
    /// The `Spring`s in the system which connect at only leaf `Mass`es.
    leaf_springs: Vec<Spring<DIM>>,
    /// The leaf `Cluster`s encountered during the evolution of the system.
    leaves_encountered: Vec<Index>,
    /// Half of the the side length of the hypercube in which the positions of
    /// the `Mass`es are initialized. The hypercube is centered at the origin.
    box_len: f32,
    /// The scaling factor of the system. This is the ratio of `box_len` to the
    /// radius of the root `Cluster`. The distances in the original dataset are
    /// scaled by this factor to find the rest lengths of the springs.
    scale: f32,
    /// The energy history of the system.
    energy_history: Vec<[f32; 2]>,
}

// The public interface of the `System` struct.
impl<'a, T: Number, C: Cluster<T>, const DIM: usize> System<'a, T, C, DIM> {
    /// Create a new `Mass`-`Spring`-`System`.
    ///
    /// # Arguments
    ///
    /// * `drag` - The drag coefficient.
    /// * `k` - The spring constant of the primary springs.
    /// * `dt` - The time-step of the simulation.
    /// * `patience` - The number of update-steps to wait before checking for
    ///   equilibrium.
    /// * `ke_threshold` - The threshold of kinetic energy below which the
    ///   system is considered to be in equilibrium.
    /// * `max_steps` - The maximum number of update-steps to run between major
    ///   updates.
    /// * `box_len` - The side length of the hypercube in which the positions of
    ///   the `Mass`es are initialized. The hypercube is centered at the origin.
    ///
    /// # Errors
    ///
    /// - If any of `drag`, `k`, `dt`, `ke_threshold`, or `box_len` is not
    ///   positive.
    /// - If any of `patience` or `max_steps` is zero.
    pub fn new(
        drag: f32,
        k: f32,
        dt: f32,
        patience: usize,
        ke_threshold: f32,
        max_steps: usize,
        box_len: f32,
    ) -> Result<Self, String> {
        if drag <= 0.0 {
            let msg = format!("The drag coefficient must be positive, but got {drag}.");
            ftlog::error!("{msg}");
            return Err(msg);
        }
        if k <= 0.0 {
            let msg = format!("The spring constant must be positive, but got {k}.");
            ftlog::error!("{msg}");
            return Err(msg);
        }
        if dt <= 0.0 {
            let msg = format!("The time-step must be positive, but got {dt}.");
            ftlog::error!("{msg}");
            return Err(msg);
        }
        if patience == 0 {
            let msg = "The patience must be positive.";
            ftlog::error!("{msg}");
            return Err(msg.to_string());
        }
        if ke_threshold <= 0.0 {
            let msg = format!("The kinetic energy threshold must be positive, but got {ke_threshold}.");
            ftlog::error!("{msg}");
            return Err(msg);
        }
        if max_steps == 0 {
            let msg = "The maximum number of steps must be positive.";
            ftlog::error!("{msg}");
            return Err(msg.to_string());
        }
        if box_len <= 0.0 {
            let msg = format!("The side length of the hypercube must be positive, but got {box_len}.");
            ftlog::error!("{msg}");
            return Err(msg);
        }

        Ok(Self {
            drag,
            k,
            dt,
            patience,
            ke_threshold,
            max_steps,
            masses: Arena::new(),
            springs: Vec::new(),
            leaf_springs: Vec::new(),
            leaves_encountered: Vec::new(),
            box_len,
            scale: 1.0,
            energy_history: Vec::new(),
        })
    }

    /// Initialize the system with a root `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `rng` - The random number generator.
    /// * `data` - The original dataset.
    /// * `metric` - The metric used to calculate the distances between the
    ///   items in the original dataset.
    /// * `root` - The root `Cluster` of the tree.
    pub fn init_with_root<I, D: Dataset<I>, M: Metric<I, T>, R: rand::Rng>(
        mut self,
        rng: &mut R,
        data: &D,
        metric: &M,
        root: &'a C,
    ) -> Self {
        self.masses = Arena::with_capacity(data.cardinality());
        self.springs.clear();
        self.leaf_springs.clear();
        self.leaves_encountered.clear();
        self.energy_history.clear();
        self.scale = self.box_len / root.radius().as_f32();

        if root.is_leaf() {
            // Add the root `Cluster` as a `Mass` at the origin.
            let m = self.add_mass(root, Vector::zero(), Vector::zero());
            self.leaves_encountered.push(m);
        } else {
            let children = root.children();
            let mut indices = Vec::new();

            // Add the children of the root `Cluster` as `Mass`es.
            for &c in &children {
                let x = Vector::random_unit(rng) * self.box_len;
                let v = Vector::random_unit(rng);
                let m = self.add_mass(c, x, v);
                indices.push(m);

                if c.is_leaf() {
                    self.leaves_encountered.push(m);
                }
            }

            // Add a `Spring` between every pair of children.
            (self.leaf_springs, self.springs) = indices
                .iter()
                .flat_map(|a| {
                    indices
                        .iter()
                        .filter(|b| self.masses[*a].arg_center() < self.masses[**b].arg_center())
                        .map(|b| (*a, *b))
                })
                .map(|(a, b)| self.new_spring(data, metric, a, b, self.k))
                .partition(|s| {
                    let [a, b] = s.mass_indices();
                    self.masses[a].is_leaf() && self.masses[b].is_leaf()
                });

            self.simulate_to_equilibrium();
        }

        self
    }

    /// Simulate the system to equilibrium and return the kinetic and potential
    /// energy of the system after the simulation.
    pub fn simulate_to_equilibrium(&mut self) -> [f32; 2] {
        let mut steps = 0;
        while steps < self.max_steps && !self.is_in_equilibrium() {
            self.update_step();
            steps += 1;
        }
        self.ke_pe()
    }

    /// Extract the positions of the `Mass`es in the system as a `FlatVec`.
    ///
    /// # Errors
    ///
    /// - If there are no `Mass`es in the system.
    pub fn extract_positions(&self) -> Result<FlatVec<[f32; DIM], usize>, String> {
        let mut positions = self
            .masses
            .iter()
            .flat_map(|(_, m)| m.indices().iter().map(|&i| (i, *m.x())).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        positions.sort_by_key(|(i, _)| *i);

        let items = positions.into_iter().map(|(_, x)| x.into()).collect::<Vec<_>>();
        FlatVec::new(items)
    }
}

// The private interface of the `System` struct.
impl<'a, T: Number, C: Cluster<T>, const DIM: usize> System<'a, T, C, DIM> {
    /// Add a `Cluster` to the system as a `Mass`.
    fn add_mass(&mut self, c: &'a C, x: Vector<DIM>, v: Vector<DIM>) -> Index {
        self.masses.insert(Mass::new(c, x, v))
    }

    /// Create a `Spring` between two `Mass`es.
    ///
    /// # Arguments
    ///
    /// * `data` - The original dataset.
    /// * `metric` - The metric used to calculate the distances between the
    ///   items in the original dataset.
    /// * `a` - The index of the first `Mass`.
    /// * `b` - The index of the second `Mass`.
    /// * `k` - The spring constant of the spring.
    #[allow(clippy::many_single_char_names)]
    fn new_spring<I, D: Dataset<I>, M: Metric<I, T>>(
        &self,
        data: &D,
        metric: &M,
        a: Index,
        b: Index,
        k: f32,
    ) -> Spring<DIM> {
        let (i, j) = (self.masses[a].arg_center(), self.masses[b].arg_center());
        let l0 = data.one_to_one(i, j, metric).as_f32() * self.scale;
        Spring::new(a, b, &self.masses, k, l0)
    }

    /// Check if the system is in equilibrium.
    fn is_in_equilibrium(&self) -> bool {
        self.energy_history.len() > self.patience && {
            let ke = self
                .energy_history
                .last()
                .unwrap_or_else(|| unreachable!("We just checked the length of the history."))[0];
            ke < self.ke_threshold
        }
    }

    /// Simulate the system for one time step and return the kinetic and
    /// potential energy of the system after the update.
    fn update_step(&mut self) -> [f32; 2] {
        let forces = self.accumulate_forces();
        for (i, f) in forces {
            self.masses[i].add_f(&f);
        }
        self.masses.iter_mut().for_each(|(_, m)| m.apply_f(self.dt, self.drag));
        self.update_springs();
        self.update_energy_history()
    }

    /// Calculate the forces exerted by the springs and accumulate them on the
    /// indices of the `Mass`es.
    fn accumulate_forces(&self) -> Vec<(Index, Vector<DIM>)> {
        let mut force_map = HashMap::new();
        for s in self.springs.iter().chain(&self.leaf_springs) {
            let [a, b] = s.mass_indices();
            let &f = s.f();
            force_map.entry(a).and_modify(|f_a| *f_a += f).or_insert(f);
            force_map.entry(b).and_modify(|f_b| *f_b -= f).or_insert(-f);
        }
        force_map.into_iter().collect()
    }

    /// Update the springs in the system.
    fn update_springs(&mut self) {
        self.springs
            .iter_mut()
            .chain(self.leaf_springs.iter_mut())
            .for_each(|s| s.recalculate(&self.masses));
    }

    /// Update the energy history of the system.
    fn update_energy_history(&mut self) -> [f32; 2] {
        let ke = self.masses.iter().map(|(_, m)| m.ke()).sum();
        let pe = self.springs.iter().map(Spring::pe).sum();
        self.energy_history.push([ke, pe]);
        [ke, pe]
    }

    /// Return the energy stored in the system.
    fn ke_pe(&self) -> [f32; 2] {
        self.energy_history.last().copied().unwrap_or([f32::MAX, f32::MAX])
    }
}
