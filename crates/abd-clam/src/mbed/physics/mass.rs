//! A `Mass` is used to represent a `Cluster` of items in the original dataset.
//!
//! `Mass`es move under forces exerted by `Spring`s and eventually come to rest,
//! representing a dimension of the dataset.

use distances::{number::Float, Number};

use crate::{Cluster, Dataset, Metric};

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
    /// Accumulator for the total magnitude of the forces experienced by the `Mass`.
    total_force_magnitude: F,
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
            force: Vector::zero(),          // Initialize force to the zero vector.
            total_force_magnitude: F::ZERO, // Initialize total force magnitude to zero.
            phantom: std::marker::PhantomData,
        }
    }

    /// Calculates the distance between  the centers of the associated
    /// `Cluster`s.
    ///
    /// # Arguments
    ///
    /// - `other`: The other `Mass` to calculate the distance to.
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculation.
    ///
    /// # Returns
    ///
    /// The distance between the two `Mass`es as a floating-point value.
    pub fn distance_between_clusters<I, D, M>(&self, other: &Self, data: &D, metric: &M) -> F
    where
        D: Dataset<I>,
        M: Metric<I, T>,
    {
        let a = self.cluster.arg_center();
        let b = other.cluster.arg_center();

        F::from(data.one_to_one(a, b, metric))
    }

    /// Calculates the euclidean distance between current positions of the
    /// `Mass`es.
    ///
    /// # Arguments
    ///
    /// - `other`: The other `Mass` to calculate the distance to.
    ///
    /// # Returns
    ///
    /// The distance between the two `Mass`es as a floating-point value.
    pub fn distance_to(&self, other: &Self) -> F {
        (self.position - other.position).magnitude()
    }

    /// Calculates the unit vector pointing from this `Mass` to another `Mass`.
    ///
    /// # Arguments
    ///
    /// - `other`: The other `Mass` to calculate the unit vector to.
    ///
    /// # Returns
    ///
    /// A `Vector` representing the unit vector pointing to the other `Mass`.
    pub fn unit_vector_to(&self, other: &Self) -> Vector<F, DIM> {
        self.position.unit_vector_to(&other.position)
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

    /// Adds a given vector to the force acting on the `Mass` and updates the total force magnitude.
    pub fn add_force(&mut self, f: &Vector<F, DIM>) {
        self.force += *f;
        self.total_force_magnitude += f.magnitude();
    }

    /// Subtracts a given vector from the force acting on the `Mass` and updates the total force magnitude.
    pub fn sub_force(&mut self, f: &Vector<F, DIM>) {
        self.force -= *f;
        self.total_force_magnitude += f.magnitude();
    }

    /// Returns the total magnitude of the forces experienced by the `Mass`.
    pub const fn total_force_magnitude(&self) -> F {
        self.total_force_magnitude
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
        self.total_force_magnitude = F::ZERO;
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
