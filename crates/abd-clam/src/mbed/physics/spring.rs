//! A `Spring` is used to preserve the relative distances between two `Mass`es
//! in the original dataset.
//!
//! `Spring`s exert forces on the `Mass`es they are connected to, pulling or
//! pushing them to maintain the relative distances between the `Mass`es when
//! we try to embed them in a lower-dimensional space.

use distances::{number::Float, Number};

use crate::{Cluster, Dataset, Metric};

use super::{
    complex::{MassKey, MassMap},
    Vector,
};

/// A `Spring` connects two masses and maintains their relative distance.
pub struct Spring<F: Float> {
    /// Keys of the two masses connected by the spring.
    keys: [MassKey; 2],
    /// The spring constant (stiffness).
    spring_constant: F,
    /// The rest length of the spring.
    rest_length: F,
    /// The current length of the spring.
    current_length: F,
    /// The magnitude of the force exerted by the spring.
    force: F,
    /// The potential energy stored in the spring.
    potential_energy: F,
}

impl<F: Float> Spring<F> {
    /// Creates a new `Spring` with the given parameters.
    ///
    /// # Arguments
    ///
    /// - `keys`: The keys of the two masses connected by the spring.
    /// - `masses`: A reference to the map of masses in the `Complex`.
    /// - `spring_constant`: The stiffness of the spring.
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculation.
    /// - `scale`: A scaling factor for the rest length of the spring.
    pub fn new<I, T, D, C, M, const DIM: usize>(
        keys: [MassKey; 2],
        masses: &MassMap<'_, T, C, F, DIM>,
        spring_constant: F,
        data: &D,
        metric: &M,
        scale: F,
    ) -> Self
    where
        T: Number,
        D: Dataset<I>,
        C: Cluster<T>,
        M: Metric<I, T>,
    {
        let [a, b] = keys;
        let [a, b] = [&masses[a], &masses[b]];
        let rest_length = a.distance_between_clusters(b, data, metric) / scale;

        let mut spring = Self {
            keys,
            spring_constant,
            rest_length,
            current_length: F::ZERO,
            force: F::ZERO,
            potential_energy: F::ZERO,
        };

        // Recalculate the current length, force, and potential energy.
        spring.recalculate(masses);

        spring
    }

    /// Returns the keys of the masses connected by the spring.
    pub const fn mass_keys(&self) -> [MassKey; 2] {
        self.keys
    }

    /// Returns the spring constant.
    pub const fn spring_constant(&self) -> F {
        self.spring_constant
    }

    /// Returns the rest length of the spring.
    pub const fn rest_length(&self) -> F {
        self.rest_length
    }

    /// Returns the current length of the spring.
    pub const fn current_length(&self) -> F {
        self.current_length
    }

    /// Returns the potential energy stored in the spring.
    pub const fn potential_energy(&self) -> F {
        self.potential_energy
    }

    /// Calculates the force exerted by the spring as a vector pointing from the
    /// first mass to the second mass.
    ///
    /// # Arguments
    ///
    /// - `masses`: A reference to the map of masses in the `Complex`.
    ///
    /// # Returns
    ///
    /// A `Vector` representing the force exerted by the spring.
    pub fn force_vector<T, C, const DIM: usize>(&self, masses: &MassMap<'_, T, C, F, DIM>) -> Vector<F, DIM>
    where
        T: Number,
        C: Cluster<T>,
    {
        let [a_key, b_key] = self.keys;
        let a = &masses[a_key];
        let b = &masses[b_key];

        let direction = a.unit_vector_to(b);
        direction * self.force
    }

    /// Recalculates the current length, force, and potential energy of the spring.
    ///
    /// # Arguments
    ///
    /// - `masses`: A reference to the map of masses in the `Complex`.
    /// - `scale`: A scaling factor for the rest length of the spring.
    pub fn recalculate<T, C, const DIM: usize>(&mut self, masses: &MassMap<'_, T, C, F, DIM>)
    where
        T: Number,
        C: Cluster<T>,
    {
        let [a_key, b_key] = self.keys;
        let a = &masses[a_key];
        let b = &masses[b_key];

        self.current_length = a.distance_to(b);
        self.force = f_mag(self.spring_constant, self.rest_length, self.current_length, 2);
        self.potential_energy = pe(self.spring_constant, self.rest_length, self.current_length, 2);
    }
}

/// Returns the magnitude of the force exerted by the spring.
///
/// The force equation is specially designed to:
///
/// - be zero when the spring is at rest,
/// - not allow the spring to bottom out under compression,
/// - produce a gently increasing force when the spring is extended, and
/// - provide a smooth transition between the two states.
///
/// # Arguments
///
/// - `k`: The spring constant (stiffness).
/// - `l0`: The rest length of the spring.
/// - `l`: The current length of the spring.
/// - `n`: The power of the reciprocal term.
fn f_mag<F: Float>(k: F, l0: F, l: F, n: i32) -> F {
    k * (l.sqrt() - l.recip().powi(n) - l0.sqrt() + l0.recip().powi(n))
}

/// Returns the potential energy stored in the spring.
///
/// The potential energy equation is the integral of the force equation.
fn pe<F: Float>(k: F, l0: F, l: F, n: i32) -> F {
    if l0 <= F::EPSILON || l.abs_diff(l0) <= F::EPSILON {
        F::ZERO
    } else {
        let pe = |x: F| {
            x * l0.recip().powi(n) - x * l0.sqrt()
                + x.powi(1 - n) / F::from(n - 1)
                + x.sqrt().cube().double() / F::from(3)
        };
        k * (pe(l) - pe(l0))
    }
}
