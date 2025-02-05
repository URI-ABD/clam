//! A point mass in the mass-spring system.

use distances::Number;
use rand::prelude::*;

use crate::{Cluster, Dataset, Metric};

/// A `Mass` in the mass-spring system for dimensionality reduction.
///
/// A `Mass` represents a `Cluster` in the reduced space, and is defined by its:
///
/// - `position`: The position of the `Mass` in the reduced space.
/// - `velocity`: The velocity of the `Mass` in the reduced space.
/// - `mass`: The mass of the `Mass`.
///
/// A `Mass` also stores state information for referencing back to the `Cluster`
/// it represents:
///
/// - `arg_center`: The index of the center of the `Cluster`.
/// - `cardinality`: The cardinality of the `Cluster`.
///
/// The `Mass` also stores the force being applied to it, which is used to
/// update the position and velocity of the `Mass`.
///
/// # Type Parameters
///
/// - `DIM`: The dimensionality of the reduced space.
/// - `T`: The type of the distance values.
/// - `S`: The type of the source `Cluster`.
#[derive(Clone, Debug)]
// pub struct Mass<const DIM: usize> {
pub struct Mass<'a, const DIM: usize, T: Number, S: Cluster<T>> {
    /// The source cluster of the `Mass`.
    source: &'a S,
    /// The position of the `Mass` in the reduced space.
    position: [f32; DIM],
    /// The velocity of the `Mass` in the reduced space.
    velocity: [f32; DIM],
    /// The force being applied to the `Mass`.
    force: [f32; DIM],
    /// The mass of the `Mass`.
    m: f32,
    /// The stress applied to the `Mass`. This is the sum of the magnitudes of
    /// the forces applied to the `Mass`.
    stress: f32,
    /// Phantom data to satisfy the type checker.
    phantom: core::marker::PhantomData<T>,
}

impl<const DIM: usize, T: Number, S: Cluster<T>> core::hash::Hash for Mass<'_, DIM, T, S> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.hash_key().hash(state);
    }
}

impl<const DIM: usize, T: Number, S: Cluster<T>> PartialEq for Mass<'_, DIM, T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.hash_key() == other.hash_key()
    }
}

impl<const DIM: usize, T: Number, S: Cluster<T>> Eq for Mass<'_, DIM, T, S> {}

impl<const DIM: usize, T: Number, S: Cluster<T>> PartialOrd for Mass<'_, DIM, T, S> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const DIM: usize, T: Number, S: Cluster<T>> Ord for Mass<'_, DIM, T, S> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.source.cmp(other.source)
    }
}

impl<'a, const DIM: usize, T: Number, S: Cluster<T>> Mass<'a, DIM, T, S> {
    /// Creates a new `Mass`.
    #[must_use]
    pub fn new(source: &'a S) -> Self {
        let m = source.cardinality().as_f32();
        Self {
            source,
            position: [0.0; DIM],
            velocity: [0.0; DIM],
            force: [0.0; DIM],
            m,
            stress: 0.0,
            phantom: core::marker::PhantomData,
        }
    }

    /// Returns the source `Cluster` of the `Mass`.
    pub const fn source(&self) -> &S {
        self.source
    }

    /// Returns a hash-key for the `Mass`.
    ///
    /// This is a 2-tuple of the `offset` and `cardinality` of the `Mass`.
    #[must_use]
    pub fn hash_key(&self) -> (usize, usize) {
        (self.arg_center(), self.cardinality())
    }

    /// Returns the index of the center of the source `Cluster`.
    #[must_use]
    pub fn arg_center(&self) -> usize {
        self.source.arg_center()
    }

    /// Returns the cardinality of the source `Cluster`.
    #[must_use]
    pub fn cardinality(&self) -> usize {
        self.source.cardinality()
    }

    /// Returns the indices of the source `Cluster`.
    #[must_use]
    pub fn indices(&self) -> Vec<usize> {
        self.source.indices()
    }

    /// Returns the mass of the `Mass`.
    #[must_use]
    pub const fn mass(&self) -> f32 {
        self.m
    }

    /// Returns the position of the `Mass`.
    #[must_use]
    pub const fn position(&self) -> &[f32; DIM] {
        &self.position
    }

    /// Returns the velocity of the `Mass`.
    #[must_use]
    pub const fn velocity(&self) -> &[f32; DIM] {
        &self.velocity
    }

    /// Returns the force being applied to the `Mass`.
    #[must_use]
    pub const fn force(&self) -> &[f32; DIM] {
        &self.force
    }

    /// Returns the stress applied to the `Mass`.
    #[must_use]
    pub const fn stress(&self) -> f32 {
        self.stress
    }

    /// Sets the mass of the `Mass` to 1.
    pub fn set_unit_mass(&mut self) {
        self.m = 1.0;
    }

    /// Returns a `Mass` with a mass of 1.
    #[must_use]
    pub fn with_unit_mass(mut self) -> Self {
        self.set_unit_mass();
        self
    }

    /// Sets the mass of the `Mass`.
    pub fn set_mass(&mut self, m: f32) {
        self.m = m;
    }

    /// Returns a `Mass` with the given mass.
    #[must_use]
    pub fn with_mass(mut self, m: f32) -> Self {
        self.set_mass(m);
        self
    }

    /// Sets the position of the `Mass`.
    pub fn set_position(&mut self, position: [f32; DIM]) {
        self.position = position;
    }

    /// Returns a `Mass` with the given position.
    #[must_use]
    pub fn with_position(mut self, position: [f32; DIM]) -> Self {
        self.set_position(position);
        self
    }

    /// Sets the velocity of the `Mass`.
    pub fn set_velocity(&mut self, velocity: [f32; DIM]) {
        self.velocity = velocity;
    }

    /// Returns a `Mass` with the given velocity.
    #[must_use]
    pub fn with_velocity(mut self, velocity: [f32; DIM]) -> Self {
        self.set_velocity(velocity);
        self
    }

    /// Resets the `velocity` of the `Mass` to the zero vector.
    pub fn reset_velocity(&mut self) {
        self.velocity = [0.0; DIM];
    }

    /// Returns the distance vector from this `Mass` to another `Mass`.
    #[must_use]
    pub fn distance_vector_to(&self, other: &Self) -> [f32; DIM] {
        let mut dv = [0.0; DIM];
        for ((d, &p), &o) in dv.iter_mut().zip(self.position.iter()).zip(other.position.iter()) {
            *d = o - p;
        }
        dv
    }

    /// Returns the current distance to another `Mass`.
    #[must_use]
    pub fn current_distance_to(&self, other: &Self) -> f32 {
        let dv = self.distance_vector_to(other);
        distances::simd::euclidean_f32(&dv, &dv)
    }

    /// Returns a unit vector pointing from this `Mass` to another `Mass`.
    #[must_use]
    pub fn unit_vector_to(&self, other: &Self) -> [f32; DIM] {
        let mut uv = self.distance_vector_to(other);

        let mag = distances::simd::euclidean_f32(&uv, &uv);
        if mag > (f32::EPSILON * DIM.as_f32()) {
            for d in &mut uv {
                *d /= mag;
            }
        } else {
            // Move the `Mass` a small distance in a random direction.
            uv = [0.0; DIM];
            let dim = rand::thread_rng().gen_range(0..DIM);
            let sign = if rand::thread_rng().gen_bool(0.5) { 1.0 } else { -1.0 };
            uv[dim] = sign;
        }
        uv
    }

    /// Adds a force to the `Mass`.
    pub fn add_force(&mut self, force: [f32; DIM]) {
        let mut f_mag = 0.0;
        for (sf, &f) in self.force.iter_mut().zip(force.iter()) {
            f_mag += f.square();
            *sf += f;
        }
        self.stress += f_mag.sqrt();
    }

    /// Subtracts a force from the `Mass`.
    pub fn sub_force(&mut self, force: [f32; DIM]) {
        let mut f_mag = 0.0;
        for (sf, &f) in self.force.iter_mut().zip(force.iter()) {
            f_mag += f.square();
            *sf -= f;
        }
        self.stress += f_mag.sqrt();
    }

    /// Applies the force to the `Mass` for one time step.
    ///
    /// This will:
    ///  - dampen the force with `beta`.
    ///  - update the velocity of the `Mass`.
    ///  - update the position of the `Mass`.
    ///  - reset the force being applied to the `Mass`.
    ///
    /// # Parameters
    ///
    /// - `dt`: The time step to apply the force for.
    /// - `beta`: The damping factor to apply to the force.
    pub fn apply_force(&mut self, dt: f32, beta: f32) {
        for ((p, v), f) in self
            .position
            .iter_mut()
            .zip(self.velocity.iter_mut())
            .zip(self.force.iter_mut())
        {
            *f -= (*v) * beta;
            *v += ((*f) / self.m) * dt;
            *p += (*v) * dt;
            *f = 0.0;
        }
    }

    /// Returns the kinetic energy of the `Mass`.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.m * distances::simd::euclidean_sq_f32(&self.velocity, &self.velocity)
    }

    /// Creates new child `Mass`es of the source `Cluster` of `self` with the
    /// same position and velocity as `self` but with forces set to zero.
    pub fn child_masses(&self) -> Vec<Self> {
        self.source
            .children()
            .iter()
            .map(|c| Self::new(c).with_position(self.position).with_velocity(self.velocity))
            .collect()
    }

    /// Creates a triangle of the `Mass` with two new `Mass`es of the children
    /// of the source `Cluster`.
    ///
    /// If the source `Cluster` has no children, then this will return `None`.
    /// Otherwise, this will return the two new `Mass`es of the children of the
    /// source `Cluster`. These two new `Mass`es will have positions set to
    /// respect the triangle formed by the three `Mass`es.
    /// This will also return the three distances forming the triangle between
    /// the three `Mass`es. These are the distances between the centers of the
    /// source `Cluster`s of `self` and `a`, `self` and `b`, and `a` and `b`
    /// respectively.
    ///
    /// The above assumes that the `metric` obeys the triangle inequality. If
    /// this is not the case, then the distances may not form a triangle. In
    /// this case, if the distances still form a triangle, then we still do the
    /// same as above. If the distances do not form a triangle, then we preserve
    /// the two shorter distances and set the longest distance to the sum of the
    /// two shorter distances.
    ///
    /// The velocities of the new `Mass`es are set to the vector of `self`.
    ///
    /// # Parameters
    ///
    /// - `dataset`: The dataset used to build the original `Cluster` tree.
    /// - `metric`: The metric to use for the `Cluster`.
    /// - `seed`: The seed to use for random number generation for selecting the
    ///   two axes along which to form the triangle.
    ///
    /// # Returns
    ///
    /// If the source `Cluster` has children, then this will return the two new
    /// `Mass`es and the three distances as described above.
    #[allow(clippy::similar_names)]
    pub fn child_triangle<I, D: Dataset<I>, M: Metric<I, T>>(
        &self,
        data: &D,
        metric: &M,
        seed: Option<u64>,
    ) -> Option<(Self, Self, T, T, T)> {
        let child_masses = self.child_masses();
        if child_masses.is_empty() {
            return None;
        }

        // Pull out the two child `Mass`es.
        let [mut a, mut b] = child_masses
            .try_into()
            .unwrap_or_else(|e: Vec<_>| unreachable!("This only works for binary trees. Got {:?} children.", e.len()));

        // Compute the distances between the `Mass`es.
        let (sa, sb, ab) = (
            self.source.distance_to(a.source, data, metric),
            self.source.distance_to(b.source, data, metric),
            a.source.distance_to(b.source, data, metric),
        );

        // Choose two random different axes along which to form the triangle.
        let (x, y) = {
            let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);
            // Sample without replacement.
            let mut axes = (0..DIM).collect::<Vec<_>>();
            axes.shuffle(&mut rng);
            (axes[0], axes[1])
        };

        // Since the positions are stores as `f32` arrays, we cast the distances
        // to `f32` for internal computations.
        let (fsa, fsb, fab) = (sa.as_f32(), sb.as_f32(), ab.as_f32());

        // Compute the deltas by which to move the child `Mass`es. Note that `a`
        // will only  be moved along the `x` axis while `b` may be moved along
        // one or both axes.
        let (dxa, dxb, dyb) = if crate::utils::is_triangle(sa, sb, ab) {
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
            let (dxa, dxb) = if crate::utils::is_colinear(sa, sb, ab) {
                if ab > sa && ab > sb {
                    // `ab` is the longest side `self` in in the middle.
                    (-fsa, fsb)
                } else if sa > ab && sa > sb {
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
                if ab > sa && ab > sb {
                    // `ab` is the longest side `self` in in the middle.
                    (-fsa, fsb)
                } else if sa > ab && sa > sb {
                    // `sa` is the longest side so `b` is in the middle.
                    (fsb + fab, fsb)
                } else {
                    // `sb` is the longest side so `a` is in the middle.
                    (fsa, fsa + fab)
                }
            };

            (dxa, dxb, 0.0)
        };

        a.position[x] += dxa;
        b.position[x] += dxb;
        b.position[y] += dyb;

        Some((a, b, sa, sb, ab))
    }
}
