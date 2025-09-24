//! A `Spring` is used to preserve the relative distances between two `Mass`es
//! in the original dataset.
//!
//! `Spring`s exert forces on the `Mass`es they are connected to, pulling or
//! pushing them to maintain the relative distances between the `Mass`es when
//! we try to embed them in a lower-dimensional space.

use crate::{Cluster, Dataset, DistanceValue, FloatDistanceValue};

use super::{
    complex::{MassKey, MassMap},
    Vector,
};

/// A `Spring` connects two masses and maintains their relative distance.
#[derive(Debug, Clone, Copy)]
pub struct Spring<F: FloatDistanceValue> {
    /// Keys of the two masses connected by the spring.
    keys: [MassKey; 2],
    /// The spring constant (stiffness).
    stiffness: F,
    /// The rest length of the spring.
    rest_length: F,
    /// The current length of the spring.
    current_length: F,
    /// The magnitude of the force exerted by the spring.
    force: F,
    /// The potential energy stored in the spring.
    potential_energy: F,
}

impl<F: FloatDistanceValue> Spring<F> {
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
    pub fn new<I, T: DistanceValue, D: Dataset<I>, C: Cluster<T>, M: Fn(&I, &I) -> T, const DIM: usize>(
        keys: [MassKey; 2],
        masses: &MassMap<'_, T, C, F, DIM>,
        stiffness: F,
        data: &D,
        metric: &M,
    ) -> Self
    where
        F: std::fmt::Debug,
    {
        let [a, b] = keys;
        let [a, b] = [&masses[a], &masses[b]];
        let rest_length = a.original_distance(b, data, metric);

        let mut spring = Self {
            keys,
            stiffness,
            rest_length,
            current_length: F::zero(),
            force: F::zero(),
            potential_energy: F::zero(),
        };

        // Recalculate the current length, force, and potential energy.
        spring.recalculate(masses);

        spring
    }

    /// Returns the keys of the masses connected by the spring.
    pub const fn mass_keys(&self) -> [MassKey; 2] {
        self.keys
    }

    /// Returns the rest length of the spring.
    pub const fn rest_length(&self) -> F {
        self.rest_length
    }

    /// Returns the potential energy stored in the spring.
    pub const fn potential_energy(&self) -> F {
        self.potential_energy
    }

    /// Returns the stiffness of the spring.
    pub const fn stiffness(&self) -> F {
        self.stiffness
    }

    /// Loosens the spring by a given factor.
    pub fn loosen(&mut self, factor: F) {
        self.stiffness *= factor;
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
        T: DistanceValue,
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
    pub fn recalculate<T: DistanceValue, C: Cluster<T>, const DIM: usize>(&mut self, masses: &MassMap<'_, T, C, F, DIM>)
    where
        F: std::fmt::Debug,
    {
        let [a_key, b_key] = self.keys;
        let a = &masses[a_key];
        let b = &masses[b_key];

        self.current_length = a.embedded_distance(b);
        debug_assert!(
            !self.current_length.is_nan(),
            "Masses {a:?} and {b:?} have NaN distance"
        );
        self.force = f_mag(self.stiffness, self.rest_length, self.current_length);
        self.potential_energy = pe(self.stiffness, self.rest_length, self.current_length);

        debug_assert!(!self.force.is_nan(), "Spring {self:?} has NaN force");
        debug_assert!(self.force.is_finite(), "Spring {self:?} has infinite force");
        debug_assert!(
            !self.potential_energy.is_nan(),
            "Spring {self:?} has NaN potential energy"
        );
        debug_assert!(
            self.potential_energy.is_finite(),
            "Spring {self:?} has infinite potential energy"
        );
    }

    /// Inherits the spring from the given parent to its children.
    pub(crate) fn inherit<I, T: DistanceValue, D: Dataset<I>, C: Cluster<T>, M: Fn(&I, &I) -> T, const DIM: usize>(
        &self,
        keys: [MassKey; 3],
        masses: &MassMap<'_, T, C, F, DIM>,
        data: &D,
        metric: &M,
        loosening_factor: F,
    ) -> [Self; 2]
    where
        F: std::fmt::Debug,
    {
        let [p, a, b] = keys;
        let o = if p == self.keys[0] { self.keys[1] } else { self.keys[0] };

        let spring_constant = self.stiffness * loosening_factor;
        let ao = Self::new([a, o], masses, spring_constant, data, metric);
        let bo = Self::new([b, o], masses, spring_constant, data, metric);

        [ao, bo]
    }
}

/// Returns the magnitude of the force exerted by the spring.
///
/// # Arguments
///
/// - `k`: The spring constant (stiffness).
/// - `l0`: The rest length of the spring.
/// - `l`: The current length of the spring.
fn f_mag<F: FloatDistanceValue>(k: F, l0: F, l: F) -> F {
    k * (l - l0).abs()
}

/// Returns the potential energy stored in the spring.
fn pe<F: FloatDistanceValue>(k: F, l0: F, l: F) -> F {
    let delta = (l - l0).abs();
    if l0 <= F::epsilon() || delta <= F::epsilon() {
        F::zero()
    } else {
        k * delta.powi(2) / (F::one() + F::one())
    }
}
