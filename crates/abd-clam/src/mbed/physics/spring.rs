//! A spring connecting two masses in the mass-spring system.

use distances::Number;

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
pub struct Spring<'a, const DIM: usize> {
    /// The first `Mass` connected by the `Spring`.
    pub(crate) a: &'a Mass<DIM>,
    /// The second `Mass` connected by the `Spring`.
    pub(crate) b: &'a Mass<DIM>,
    /// The spring constant of the `Spring`.
    k: f32,
    /// The length of the `Spring` in the original embedding space cast to `f32`.
    l0: f32,
    /// The length of the `Spring` in the reduced space.
    l: f32,
    /// The magnitude force exerted by the `Spring`.
    f_mag: f32,
}

impl<'a, const DIM: usize> Spring<'a, DIM> {
    /// Create a new `Spring`.
    pub fn new<T: Number>(a: &'a Mass<DIM>, b: &'a Mass<DIM>, k: f32, l0: T) -> Self {
        let mut s = Self {
            a,
            b,
            k,
            l0: l0.as_f32(),
            l: l0.as_f32(),
            f_mag: 0.0,
        };
        s.update_force();
        s
    }

    /// Get the hash key of the `Spring`.
    ///
    /// The hash key is a tuple of the hash keys of the two connected `Mass`es.
    /// This is used to uniquely identify the `Spring` in the `System`.
    pub const fn hash_key(&self) -> ((usize, usize), (usize, usize)) {
        (self.a.hash_key(), self.b.hash_key())
    }

    /// Get the rest length of the `Spring`.
    pub const fn l0(&self) -> f32 {
        self.l0
    }

    /// Get the spring constant of the `Spring`.
    pub const fn k(&self) -> f32 {
        self.k
    }

    /// Get the length of the `Spring`.
    pub const fn l(&self) -> f32 {
        self.l
    }

    /// Get the displacement magnitude of the `Spring`.
    pub fn dx(&self) -> f32 {
        self.l0 - self.l
    }

    /// Get the magnitude of the force exerted by the `Spring`.
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

impl<const DIM: usize> core::hash::Hash for Spring<'_, DIM> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.hash_key().hash(state);
    }
}

impl<const DIM: usize> PartialEq for Spring<'_, DIM> {
    fn eq(&self, other: &Self) -> bool {
        self.hash_key() == other.hash_key()
    }
}

impl<const DIM: usize> Eq for Spring<'_, DIM> {}
