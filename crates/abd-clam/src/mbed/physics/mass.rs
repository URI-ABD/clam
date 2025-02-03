//! A point mass in the mass-spring system.

use distances::Number;
use rand::prelude::*;

use crate::Cluster;

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
            phantom: core::marker::PhantomData,
        }
    }

    /// Returns a hash-key for the `Mass`.
    ///
    /// This is a 2-tuple of the `offset` and `cardinality` of the `Mass`.
    #[must_use]
    pub fn hash_key(&self) -> (usize, usize) {
        (self.arg_center(), self.cardinality())
    }

    /// Returns the index of the center of the `Cluster`.
    #[must_use]
    pub fn arg_center(&self) -> usize {
        self.source.arg_center()
    }

    /// Returns the cardinality of the `Cluster`.
    #[must_use]
    pub fn cardinality(&self) -> usize {
        self.source.cardinality()
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
        for (sf, &f) in self.force.iter_mut().zip(force.iter()) {
            *sf += f;
        }
    }

    /// Subtracts a force from the `Mass`.
    pub fn sub_force(&mut self, force: [f32; DIM]) {
        for (sf, &f) in self.force.iter_mut().zip(force.iter()) {
            *sf -= f;
        }
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
}
