//! A spring connecting two masses in the mass-spring system.

use distances::Number;

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
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)
)]
pub struct Spring {
    /// The hash of the first `Mass` connected by the `Spring`.
    a_key: (usize, usize),
    /// The hash of the second `Mass` connected by the `Spring`.
    b_key: (usize, usize),
    /// The spring constant of the `Spring`.
    k: f32,
    /// The length of the `Spring` in the original embedding space cast to `f32`.
    l0: f32,
    /// The length of the `Spring` in the reduced space.
    l: f32,
    /// The magnitude force exerted by the `Spring`.
    f_mag: f32,
}

impl core::hash::Hash for Spring {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.hash_key().hash(state);
    }
}

impl PartialEq for Spring {
    fn eq(&self, other: &Self) -> bool {
        self.hash_key() == other.hash_key()
    }
}

impl Eq for Spring {}

impl Spring {
    /// Create a new `Spring`.
    pub fn new<T: Number>(a_key: (usize, usize), b_key: (usize, usize), k: f32, l0: T, l: f32) -> Self {
        let mut s = Self {
            a_key,
            b_key,
            k,
            l0: l0.as_f32(),
            l,
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
        (self.a_key, self.b_key)
    }

    /// Returns the hash key of the first `Mass` connected by the `Spring`.
    pub const fn a_key(&self) -> (usize, usize) {
        self.a_key
    }

    /// Returns the hash key of the second `Mass` connected by the `Spring`.
    pub const fn b_key(&self) -> (usize, usize) {
        self.b_key
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
            a_key: self.a_key,
            b_key: self.b_key,
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
    pub fn connects(&self, m_key: (usize, usize)) -> bool {
        self.a_key == m_key || self.b_key == m_key
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
    pub fn update_length(&mut self, l: f32) {
        // self.l = self.a.current_distance_to(self.b);
        self.l = l;
        self.update_force();
    }

    /// Get the potential energy of the `Spring`.
    pub fn potential_energy(&self) -> f32 {
        0.5 * self.k * self.dx().powi(2)
    }
}
