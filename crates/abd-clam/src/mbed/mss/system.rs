//! The `Mass`-`Spring`-`System`.

use std::collections::HashMap;

use distances::Number;
use generational_arena::{Arena, Index};
use rand::prelude::*;

use crate::{mbed::Vector, Cluster, Dataset, FlatVec, Metric, SizedHeap};

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
    /// The multiplicative factor by which the springs are loosened when they
    /// are inherited by the children of a parent mass.
    loosening_factor: f32,
    /// The fraction of the most stressed masses to replace with child masses at
    /// the beginning of a major update.
    replace_fraction: f32,
    /// The number of times a spring can be loosened before it is removed.
    loosening_threshold: usize,
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
    /// * `loosening_factor` - The multiplicative factor by which the springs
    ///   are loosened when they are inherited by the children of a parent mass.
    /// * `replace_fraction` - The fraction of the most stressed masses to
    ///   replace with child masses at the beginning of a major update.
    /// * `loosening_threshold` - The number of times a spring can be loosened
    ///   before it is removed.
    ///
    /// # Errors
    ///
    /// - If any of `drag`, `k`, `dt`, `ke_threshold` or `box_len` is not
    ///   positive.
    /// - If any of `patience` or `max_steps` is zero.
    /// - If `loosening_factor` is not in the range `(0, 1)`.
    /// - If `replace_fraction` is not in the range `(0, 1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        drag: f32,
        k: f32,
        dt: f32,
        patience: usize,
        ke_threshold: f32,
        max_steps: usize,
        box_len: f32,
        loosening_factor: f32,
        replace_fraction: f32,
        loosening_threshold: usize,
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
        if !(0.0..1.0).contains(&loosening_factor) {
            let msg = format!("The loosening factor must be in the range (0, 1), but got {loosening_factor}.");
            ftlog::error!("{msg}");
            return Err(msg);
        }
        if !(0.0..=1.0).contains(&replace_fraction) {
            let msg = format!("The replace fraction must be in the range (0, 1], but got {replace_fraction}.");
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
            loosening_factor,
            replace_fraction,
            loosening_threshold,
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
            self.add_mass(Mass::new(root, Vector::zero(), Vector::zero()));
        } else {
            let children = root.children();
            let mut indices = Vec::new();

            // Add the children of the root `Cluster` as `Mass`es.
            for &c in &children {
                let r = rng.gen_range(0.0..1.0);
                let len = r * self.box_len;
                let x = Vector::random(rng, -len, len);

                let r = rng.gen_range(0.0..1.0);
                let len = r * self.box_len;
                let v = Vector::random(rng, -len, len);

                indices.push(self.add_mass(Mass::new(c, x, v)));
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
                .map(|(a, b)| self.new_spring(data, metric, a, b, self.k, 0))
                .partition(Spring::is_leaf_spring);

            self.simulate_to_equilibrium();
        }

        self
    }

    /// Simulate the system until it reaches equilibrium with only the leaf
    /// `Cluster`s being represented as `Mass`es.
    pub fn simulate_to_leaves<I, D: Dataset<I>, M: Metric<I, T>, R: rand::Rng>(
        &mut self,
        rng: &mut R,
        data: &D,
        metric: &M,
    ) -> [f32; 2] {
        let mut step = 0;
        let [mut ke, mut pe] = self.update_energy_history();

        // Repeat until only leaf masses are left.
        while !self.springs.is_empty() {
            step += 1;
            ftlog::info!(
                "Starting major update {step} with {} masses, {} leaves, {} springs and {} leaf springs...",
                self.masses.len(),
                self.leaves_encountered.len(),
                self.springs.len(),
                self.leaf_springs.len()
            );
            self.log_system();

            // Remove the weak springs.
            let _ = self.remove_weak_springs();

            // Find the most stressed masses to be replaced.
            let stressed_masses = self.most_stressed_parent_masses();
            ftlog::debug!("Stressed masses: {stressed_masses:?}");

            // For each stressed mass, generate a pair of random unit vectors
            // along which its child clusters will be moved.
            let xy_pairs = stressed_masses
                .iter()
                .map(|_| {
                    let x = Vector::<DIM>::random_unit(rng);
                    let y = x.perpendicular(rng);
                    [x, y]
                })
                .collect::<Vec<_>>();

            // For each such mass, make a triangle of springs with its two
            // children at the other vertices.
            #[allow(clippy::needless_collect)]
            let triangles = stressed_masses
                .into_iter()
                .zip(xy_pairs)
                .map(|(i, [x, y])| {
                    let m = &self.masses[i];
                    (i, m.child_triangle(data, metric, x, y, self.scale))
                })
                .collect::<Vec<_>>();
            ftlog::debug!("Triangles: {triangles:?}");

            // Add the new masses to the system.
            let triangles = triangles
                .into_iter()
                .map(|(i, [a, b])| (i, [self.add_mass(a), self.add_mass(b)]))
                .collect::<HashMap<_, _>>();
            let masses = self.masses.iter().collect::<Vec<_>>();
            ftlog::debug!("New Masses: {masses:?}");

            // Add springs for the sides of the triangles.
            let (new_leaf_springs, new_springs): (Vec<_>, _) = triangles
                .iter()
                .flat_map(|(&c, &[a, b])| {
                    let ca = self.new_spring(data, metric, c, a, self.k, 0);
                    let cb = self.new_spring(data, metric, c, b, self.k, 0);
                    let ab = self.new_spring(data, metric, a, b, self.k, 0);
                    [ca, cb, ab]
                })
                .partition(Spring::is_leaf_spring);
            ftlog::debug!("New springs: {new_springs:?}");
            ftlog::debug!("New leaf springs: {new_leaf_springs:?}");
            self.springs.extend(new_springs);
            self.leaf_springs.extend(new_leaf_springs);

            // The children inherit springs from the parent, loosening the
            // springs in the process.
            let (new_leaf_springs, new_springs): (Vec<_>, _) = triangles
                .iter()
                .flat_map(|(c, [a, b])| {
                    self.springs_of(*c)
                        .into_iter()
                        .filter(|(d, _)| self.masses[*c].arg_center() < self.masses[*d].arg_center())
                        .flat_map(|(d, cd)| {
                            // The spring between c and d will be loosened as it
                            // is inherited by the children of c.
                            let times_loosened = cd.times_loosened() + 1;
                            let k = cd.k() * self.loosening_factor;

                            if let Some([e, f]) = triangles.get(&d) {
                                // d is being replaced, so we will add four new springs.
                                let ae = self.new_spring(data, metric, *a, *e, k, times_loosened);
                                let af = self.new_spring(data, metric, *a, *f, k, times_loosened);
                                let be = self.new_spring(data, metric, *b, *e, k, times_loosened);
                                let bf = self.new_spring(data, metric, *b, *f, k, times_loosened);
                                vec![ae, af, be, bf]
                            } else {
                                // d is not being replaced, so we will add two new springs.
                                let da = self.new_spring(data, metric, d, *a, k, times_loosened);
                                let db = self.new_spring(data, metric, d, *b, k, times_loosened);
                                vec![da, db]
                            }
                        })
                })
                .partition(Spring::is_leaf_spring);
            self.springs.extend(new_springs);
            self.leaf_springs.extend(new_leaf_springs);

            // Simulate the system to equilibrium.
            self.simulate_to_equilibrium();

            // Remove the parent masses and all springs connected to them.
            for c in triangles.keys() {
                self.masses.remove(*c);
                self.springs.retain(|s| !s.connects(*c));
            }
            self.springs.retain(|s| s.is_ordered(&self.masses));

            // Simulate the system to equilibrium.
            [ke, pe] = self.simulate_to_equilibrium();

            // Loosen all leaf springs.
            self.leaf_springs
                .iter_mut()
                .for_each(|s| s.loosen(self.loosening_factor));

            // Add leaf springs among a random subset of the leaf masses.
            self.add_random_leaf_springs(rng, data, metric);

            ftlog::info!("Major update {step} complete.");
            self.log_system();
            if step > 2 {
                unimplemented!();
            }
        }

        [ke, pe]
    }

    /// Simulate the system to equilibrium and return the kinetic and potential
    /// energy of the system after the simulation.
    pub fn simulate_to_equilibrium(&mut self) -> [f32; 2] {
        let mut step = 0;
        let [mut ke, mut pe] = [f32::MAX, f32::MAX];

        while step < self.max_steps && !self.is_in_equilibrium() {
            step += 1;
            ftlog::trace!("Starting minor update {step}...");
            [ke, pe] = self.update_step();
            ftlog::trace!("Minor update {step} complete. KE: {ke:.2e}, PE: {pe:.2e}");
        }

        [ke, pe]
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
    /// Log the system.
    pub fn log_system(&self) {
        self.log_masses();
        self.log_springs();
        self.log_leaf_springs();
    }

    /// Log the masses in the system.
    fn log_masses(&self) {
        if !self.masses.is_empty() {
            ftlog::debug!("Masses: ");
            self.masses.iter().for_each(|(i, m)| ftlog::debug!("{i:?}: {m:?}"));
        }
    }

    /// Log the springs in the system.
    fn log_springs(&self) {
        if !self.springs.is_empty() {
            ftlog::debug!("Springs: ");
            self.springs.iter().for_each(|s| ftlog::debug!("{s:?}"));
        }
    }

    /// Log the leaf springs in the system.
    fn log_leaf_springs(&self) {
        if !self.leaf_springs.is_empty() {
            ftlog::debug!("Leaf springs: ");
            self.leaf_springs.iter().for_each(|s| ftlog::debug!("{s:?}"));
        }
    }

    /// Add a `Cluster` to the system as a `Mass`.
    fn add_mass(&mut self, m: Mass<'a, T, C, DIM>) -> Index {
        if m.is_leaf() {
            let m = self.masses.insert(m);
            self.leaves_encountered.push(m);
            m
        } else {
            self.masses.insert(m)
        }
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
        times_loosened: usize,
    ) -> Spring<DIM> {
        let (i, j) = (&self.masses[a], &self.masses[b]);
        let connects_leaves = i.is_leaf() && j.is_leaf();

        let (i, j) = (i.arg_center(), j.arg_center());
        let l0 = data.one_to_one(i, j, metric).as_f32() * self.scale;

        Spring::new(a, b, &self.masses, k, l0, times_loosened, connects_leaves)
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
        let pe = self
            .springs
            .iter()
            .chain(self.leaf_springs.iter())
            .map(Spring::pe)
            .sum();
        self.energy_history.push([ke, pe]);
        [ke, pe]
    }

    /// Remove the weak springs from the system.
    fn remove_weak_springs(&mut self) -> [Vec<Spring<DIM>>; 2] {
        let loose_springs;
        (loose_springs, self.springs) = self
            .springs
            .drain(..)
            .partition(|s| s.is_too_loose(self.loosening_threshold));

        let loose_leaf_springs;
        (loose_leaf_springs, self.leaf_springs) = self
            .leaf_springs
            .drain(..)
            .partition(|s| s.is_too_loose(self.loosening_threshold));

        [loose_springs, loose_leaf_springs]
    }

    /// Return the indices of the `Mass`es sorted by the sum of the magnitudes
    /// of the forces acting on them from the springs.
    fn most_stressed_parent_masses(&self) -> Vec<Index> {
        let heap = self
            .masses
            .iter()
            .filter_map(|(i, m)| if m.is_leaf() { None } else { Some((m.f_mag(), i)) })
            .collect::<SizedHeap<_>>();
        let num_masses = (self.replace_fraction * heap.len().as_f32()).ceil().as_usize();
        heap.items().take(num_masses).map(|(_, i)| i).collect()
    }

    /// Returns the neighbors of the `Mass` by its index in the arena.
    fn springs_of(&self, i: Index) -> Vec<(Index, &Spring<DIM>)> {
        let v = self
            .springs
            .iter()
            .filter_map(|s| s.neighbor_of(i).map(|j| (j, s)))
            .collect::<Vec<_>>();
        ftlog::trace!("Springs of {i:?}: {v:?}");
        v
    }

    /// Adds leaf springs among a random subset of the leaf masses.
    fn add_random_leaf_springs<I, D: Dataset<I>, M: Metric<I, T>, R: rand::Rng>(
        &mut self,
        rng: &mut R,
        data: &D,
        metric: &M,
    ) {
        let pairs = {
            let mut pairs = self
                .leaves_encountered
                .iter()
                .enumerate()
                .flat_map(|(i, &a)| self.leaves_encountered.iter().skip(i + 1).map(move |&b| (a, b)))
                .collect::<Vec<_>>();
            pairs.shuffle(rng);

            let n_pairs = ((1.0 - self.replace_fraction) * self.leaves_encountered.len().as_f32())
                .square()
                .ceil()
                .as_usize();
            pairs.truncate(n_pairs);

            pairs
        };

        if !pairs.is_empty() {
            let new_springs = pairs
                .into_iter()
                .map(|(a, b)| self.new_spring(data, metric, a, b, self.k, 0))
                .collect::<Vec<_>>();
            self.leaf_springs.extend(new_springs);
        }
    }
}
