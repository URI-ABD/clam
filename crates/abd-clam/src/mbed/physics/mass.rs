//! A `Mass` is used to represent a `Cluster` of items in the original dataset.
//!
//! `Mass`es move under forces exerted by `Spring`s and eventually come to rest,
//! representing a dimension of the dataset.

use distances::{number::Float, Number};

use crate::Cluster;

use super::Vector;

/// Represents a `Mass` in the system.
pub struct Mass<'a, T: Number, C: Cluster<T>, F: Float, const DIM: usize> {
    /// Reference to the associated `Cluster`.
    cluster: &'a C,
    /// The position of the `Mass` in the system.
    position: Vector<F, DIM>,
    /// The velocity of the `Mass` in the system.
    velocity: Vector<F, DIM>,
    /// The force currently acting on the `Mass`.
    force: Vector<F, DIM>,
    /// Phantom data to store the type `T`.
    phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Number, C: Cluster<T>, F: Float, const DIM: usize> Mass<'a, T, C, F, DIM> {
    /// Creates a new `Mass`.
    ///
    /// # Arguments
    ///
    /// - `cluster`: A reference to the associated `Cluster`.
    /// - `position`: The initial position of the `Mass`.
    /// - `velocity`: The initial velocity of the `Mass`.
    pub const fn new(cluster: &'a C, position: Vector<F, DIM>, velocity: Vector<F, DIM>) -> Self {
        Self {
            cluster,
            position,
            velocity,
            force: Vector::zero(), // Initialize force to the zero vector.
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns a reference to the associated `Cluster`.
    pub const fn cluster(&self) -> &C {
        self.cluster
    }

    /// Returns a reference to the position of the `Mass`.
    pub const fn position(&self) -> &Vector<F, DIM> {
        &self.position
    }

    /// Returns a reference to the velocity of the `Mass`.
    pub const fn velocity(&self) -> &Vector<F, DIM> {
        &self.velocity
    }

    /// Returns a reference to the force acting on the `Mass`.
    pub const fn force(&self) -> &Vector<F, DIM> {
        &self.force
    }

    /// Returns the mass of the `Mass`, which is equal to the cardinality of the cluster.
    pub fn mass(&self) -> F {
        F::from(self.cluster.cardinality())
    }

    /// Adds a given vector to the force acting on the `Mass`.
    pub fn add_force(&mut self, f: &Vector<F, DIM>) {
        self.force += *f;
    }

    /// Subtracts a given vector from the force acting on the `Mass`.
    pub fn sub_force(&mut self, f: &Vector<F, DIM>) {
        self.force -= *f;
    }

    /// Moves the `Mass` under the force it experiences.
    ///
    /// # Arguments
    ///
    /// - `drag`: The drag coefficient to reduce velocity.
    /// - `dt`: The size of the time step for the movement.
    pub fn move_mass(&mut self, drag: F, dt: F) {
        // Calculate drag force.
        let friction = -self.velocity * drag;

        // Calculate acceleration based on force and mass.
        let acceleration = (self.force + friction) / self.mass();

        // Update velocity based on acceleration and drag force.
        self.velocity += acceleration * dt;

        // Update position based on velocity.
        self.position += self.velocity * dt;

        // Reset the force to the zero vector.
        self.force = Vector::zero();
    }

    /// Calculates the kinetic energy of the `Mass`.
    ///
    /// # Returns
    ///
    /// The kinetic energy as a floating-point value.
    pub fn kinetic_energy(&self) -> F {
        let speed_squared = self.velocity.dot(&self.velocity);
        self.mass() * speed_squared.half()
    }
}
