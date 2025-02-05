//! A spring connecting two masses in the mass-spring system.

use distances::Number;

use crate::Cluster;

use super::Mass;

/// A spring  connecting two masses in the mass-spring system.
///
/// The spring is defined by its:
///
/// - spring constant `k`, i.e. the stiffness of the `Spring`,
/// - rest length `l0`, i.e. the distance between the two connected `Cluster`s in the original embedding space.
/// - current length `l`, i.e. the distance between the two connected `Mass`es in the reduced space.
///
/// # Type Parameters
///
/// - `DIM`: The dimensionality of the reduced space.
pub struct Spring<'a, const DIM: usize, T: Number, S: Cluster<T>> {
    /// The first `Mass` connected by the `Spring`.
    a: &'a Mass<'a, DIM, T, S>,
    /// The second `Mass` connected by the `Spring`.
    b: &'a Mass<'a, DIM, T, S>,
    /// The spring constant of the `Spring`.
    k: f32,
    /// The length of the `Spring` in the original embedding space cast to `f32`.
    l0: f32,
    /// The length of the `Spring` in the reduced space.
    l: f32,
    /// The magnitude force exerted by the `Spring`.
    f_mag: f32,
}

impl<const DIM: usize, T: Number, S: Cluster<T>> core::hash::Hash for Spring<'_, DIM, T, S> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.hash_key().hash(state);
    }
}

impl<const DIM: usize, T: Number, S: Cluster<T>> PartialEq for Spring<'_, DIM, T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.hash_key() == other.hash_key()
    }
}

impl<const DIM: usize, T: Number, S: Cluster<T>> Eq for Spring<'_, DIM, T, S> {}

impl<'a, const DIM: usize, T: Number, S: Cluster<T>> Spring<'a, DIM, T, S> {
    /// Create a new `Spring`.
    pub fn new(a: &'a Mass<DIM, T, S>, b: &'a Mass<DIM, T, S>, k: f32, l0: T) -> Self {
        let mut s = Self {
            a,
            b,
            k,
            l0: l0.as_f32(),
            l: a.current_distance_to(b),
            f_mag: 0.0,
        };
        s.update_force();
        s
    }

    /// Get the hash key of the `Spring`.
    ///
    /// The hash key is a tuple of the hash keys of the two connected `Mass`es.
    /// This is used to uniquely identify the `Spring` in the `System`.
    pub fn hash_key(&self) -> ((usize, usize), (usize, usize)) {
        (self.a.hash_key(), self.b.hash_key())
    }

    /// Get the first `Mass` connected by the `Spring`.
    pub const fn a(&self) -> &'a Mass<DIM, T, S> {
        self.a
    }

    /// Returns the hash key of the first `Mass` connected by the `Spring`.
    pub fn a_key(&self) -> (usize, usize) {
        self.a.hash_key()
    }

    /// Returns the hash key of the second `Mass` connected by the `Spring`.
    pub fn b_key(&self) -> (usize, usize) {
        self.b.hash_key()
    }

    /// Get the second `Mass` connected by the `Spring`.
    pub const fn b(&self) -> &'a Mass<DIM, T, S> {
        self.b
    }

    /// Get the rest length of the `Spring`.
    pub const fn l0(&self) -> f32 {
        self.l0
    }

    /// Get the spring constant of the `Spring`.
    pub const fn k(&self) -> f32 {
        self.k
    }

    /// Return a copy of the `Spring` with the given spring constant.
    pub const fn with_k(&self, k: f32) -> Self {
        Self {
            a: self.a,
            b: self.b,
            k,
            l0: self.l0,
            l: self.l,
            f_mag: self.f_mag,
        }
    }

    /// Get the length of the `Spring`.
    pub const fn l(&self) -> f32 {
        self.l
    }

    /// Return whether the `Spring` connects the given `Mass`.
    pub fn connects(&self, m: &Mass<DIM, T, S>) -> bool {
        self.a == m || self.b == m
    }

    /// Get the displacement of the `Spring` from its rest length.
    pub fn dx(&self) -> f32 {
        self.l0 - self.l
    }

    /// Get the magnitude of the force exerted by the `Spring`.
    ///
    /// If the force is somehow not finite, return `1.0` instead.
    pub fn f_mag(&self) -> f32 {
        if self.f_mag.is_finite() {
            self.f_mag
        } else {
            1.0
        }
    }

    /// Update the force exerted by the `Spring`.
    pub fn update_force(&mut self) {
        self.f_mag = -self.k * self.dx();
    }

    /// Update the length and force of the `Spring`.
    pub fn update_length(&mut self) {
        self.l = self.a.current_distance_to(self.b);
        self.update_force();
    }

    /// Get the potential energy of the `Spring`.
    pub fn potential_energy(&self) -> f32 {
        0.5 * self.k * self.dx().powi(2)
    }
}
