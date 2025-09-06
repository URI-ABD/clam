//! A `Mass` is used to represent a `Cluster` of items in the original dataset.
//!
//! `Mass`es move under forces exerted by `Spring`s and eventually come to rest,
//! representing a dimension of the dataset.

use std::sync::{Arc, RwLock};

use rand::{distr::uniform::SampleUniform, Rng};

use crate::{Cluster, Dataset, DistanceValue, FloatDistanceValue};

use super::Vector;

/// Represents a `Mass` in the system.
#[derive(Clone)]
pub struct Mass<'a, T: DistanceValue, C: Cluster<T>, F: FloatDistanceValue, const DIM: usize> {
    /// Reference to the associated `Cluster`.
    cluster: &'a C,
    /// The position of the `Mass` in the system.
    position: Vector<F, DIM>,
    /// The velocity of the `Mass` in the system.
    velocity: Vector<F, DIM>,
    /// The forces currently acting on the `Mass`.
    forces: Arc<RwLock<Vec<Vector<F, DIM>>>>,
    /// The total force Vector acting on the `Mass`.
    total_force: Vector<F, DIM>,
    /// Phantom data to store the type `T`.
    phantom: std::marker::PhantomData<T>,
}

impl<T: DistanceValue, C: Cluster<T>, F: FloatDistanceValue + std::fmt::Debug, const DIM: usize> std::fmt::Debug
    for Mass<'_, T, C, F, DIM>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mass")
            .field("position", &self.position)
            .field("velocity", &self.velocity)
            .field("total_force", &self.total_force)
            .finish()
    }
}

impl<'a, T: DistanceValue, C: Cluster<T>, F: FloatDistanceValue, const DIM: usize> Mass<'a, T, C, F, DIM> {
    /// Creates a new `Mass`.
    ///
    /// # Arguments
    ///
    /// - `cluster`: A reference to the associated `Cluster`.
    /// - `position`: The initial position of the `Mass`.
    /// - `velocity`: The initial velocity of the `Mass`.
    pub fn new(cluster: &'a C, position: Vector<F, DIM>, velocity: Vector<F, DIM>) -> Self {
        Self {
            cluster,
            position,
            velocity,
            forces: Arc::new(RwLock::new(Vec::new())),
            total_force: Vector::zero(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns the position of the `Mass`.
    pub const fn position(&self) -> Vector<F, DIM> {
        self.position
    }

    /// Shifts the position of the `Mass` by a given vector.
    pub fn shift_position(&mut self, shift: Vector<F, DIM>) {
        self.position += shift;
    }

    /// Sets a velocity for the `Mass`.
    pub const fn set_velocity(&mut self, velocity: Vector<F, DIM>) {
        self.velocity = velocity;
    }

    /// Calculates the distance in the original dataset.
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
    pub fn original_distance<I, D, M>(&self, other: &Self, data: &D, metric: &M) -> F
    where
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
    {
        let a = self.cluster.arg_center();
        let b = other.cluster.arg_center();

        F::from(data.one_to_one(a, b, metric)).unwrap_or_else(F::infinity)
    }

    /// Calculates the euclidean distance in the embedding space.
    ///
    /// # Arguments
    ///
    /// - `other`: The other `Mass` to calculate the distance to.
    ///
    /// # Returns
    ///
    /// The distance between the two `Mass`es as a floating-point value.
    pub fn embedded_distance(&self, other: &Self) -> F {
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

    /// Returns whether the `Mass` represents a leaf cluster.
    pub fn is_leaf(&self) -> bool {
        self.cluster.is_leaf()
    }

    /// Returns the radius of the cluster represented by the `Mass`.
    pub fn radius(&self) -> F {
        F::from(self.cluster.radius()).unwrap_or_else(F::infinity)
    }

    /// Returns the mass of the `Mass`, which is equal to the cardinality of the cluster.
    pub fn mass(&self) -> F {
        F::from(self.cluster.cardinality()).unwrap_or_else(|| unreachable!("Cluster cardinality overflowed F"))
    }

    /// Adds a given vector to the force acting on the `Mass`.
    ///
    /// # Panics
    ///
    /// - If the `forces` lock is poisoned.
    #[allow(clippy::unwrap_used)]
    pub fn add_force(&self, f: &Vector<F, DIM>) {
        self.forces.write().unwrap().push(*f);
    }

    /// Subtracts a given vector from the force acting on the `Mass`.
    ///
    /// # Panics
    ///
    /// - If the `forces` lock is poisoned.
    #[allow(clippy::unwrap_used)]
    pub fn sub_force(&self, f: &Vector<F, DIM>) {
        self.forces.write().unwrap().push(-*f);
    }

    /// Moves the `Mass` under the force it experiences.
    ///
    /// # Arguments
    ///
    /// - `drag`: The drag coefficient to reduce velocity.
    /// - `dt`: The size of the time step for the movement.
    ///
    /// # Panics
    ///
    /// - If the `forces` lock is poisoned.
    #[allow(clippy::unwrap_used)]
    pub fn move_mass(&mut self, drag: F, dt: F)
    where
        F: std::fmt::Debug,
    {
        // Accumulate all forces acting on the mass.
        self.total_force = self.forces.write().unwrap().drain(..).sum();

        // Calculate friction from drag.
        let friction = -self.velocity * drag;

        // Calculate acceleration based on force and mass.
        let acceleration = (self.total_force + friction) / self.mass();

        // Update velocity based on acceleration.
        self.velocity += acceleration * dt;

        debug_assert!(!self.velocity.has_nan(), "{self:?} has NaN velocity");
        debug_assert!(!self.velocity.has_inf(), "{self:?} has Inf velocity");

        // Update position based on velocity.
        self.position += self.velocity * dt;

        debug_assert!(!self.position.has_nan(), "{self:?} has NaN position");
        debug_assert!(!self.position.has_inf(), "{self:?} has Inf position");
    }

    /// Calculates the kinetic energy of the `Mass`.
    ///
    /// # Returns
    ///
    /// The kinetic energy as a floating-point value.
    pub fn kinetic_energy(&self) -> F {
        let speed_squared = self.velocity.dot(&self.velocity);
        self.mass() * speed_squared / (F::one() + F::one())
    }

    /// Returns the indices and positions of the items in the `Cluster`.
    pub fn itemized_positions(&self) -> Vec<(usize, Vector<F, DIM>)> {
        let mut positions = self
            .cluster
            .indices()
            .iter()
            .map(|&i| (i, self.position))
            .collect::<Vec<_>>();
        // sort the positions by index
        positions.sort_by_key(|(i, _)| *i);
        positions
    }

    /// Explodes the `Mass`, creating new masses for the children of the cluster it represents.
    ///
    /// The caller must ensure that the `Mass` does not represent a leaf cluster.
    ///
    /// # Arguments
    ///
    /// - `rng`: A reference to the random number generator.
    /// - `drag`: The drag coefficient to reduce velocity.
    /// - `dt`: The size of the time step for the movement.
    ///
    /// # Returns
    ///
    /// A vector of new `Mass`es representing the children of the cluster.
    ///
    /// # Panics
    ///
    /// - If the `Mass` represents a leaf cluster.
    #[allow(clippy::panic)]
    pub(crate) fn explode<R: Rng>(&self, rng: &mut R, drag: F, dt: F) -> [Self; 2]
    where
        F: SampleUniform + std::fmt::Debug,
    {
        debug_assert!(!self.cluster.is_leaf(), "Cannot explode a leaf cluster.");

        // There should be exactly two children.
        let [a, b] = {
            let children = self.cluster.children();
            [children[0], children[1]]
        };
        let [ma, mb] = [
            F::from(a.cardinality()).unwrap_or_else(|| unreachable!("Cluster cardinality overflowed F")),
            F::from(b.cardinality()).unwrap_or_else(|| unreachable!("Cluster cardinality overflowed F")),
        ];

        let v_mag = self.velocity.magnitude();
        let va = if v_mag > F::epsilon() {
            // The first child will have a new velocity in a random direction and
            // its magnitude will be proportional to its mass.
            Vector::random_unit(rng) * (v_mag * ma / self.mass())
        } else {
            // The mass is at rest, so the first child will have a random velocity.
            Vector::random_unit(rng)
        };

        // Use conservation of momentum to calculate the velocity of the second child.
        let vb = (self.velocity * self.mass() - va * ma) / mb;

        // Create the new masses.
        let mut a = Mass::new(a, self.position, va);
        let mut b = Mass::new(b, self.position, vb);

        debug_assert!(!va.has_nan(), "Child {a:?} of {self:?} has NaN velocity");
        debug_assert!(!va.has_inf(), "Child {a:?} of {self:?} has Inf velocity");
        debug_assert!(!vb.has_nan(), "Child {b:?} of {self:?} has NaN velocity");
        debug_assert!(!vb.has_inf(), "Child {b:?} of {self:?} has Inf velocity");

        // Move the masses due to the explosion.
        a.move_mass(drag, dt);
        b.move_mass(drag, dt);

        [a, b]
    }
}
