//! A `Spring` between two `Cluster`s in a mass-spring system.

use distances::Number;

use crate::Cluster;

use super::{System, Vector};

/// A spring connecting two `Cluster`s in a mass-spring system.
#[derive(Debug, PartialEq)]
pub struct Spring<'a, T: Number, C: Cluster<T>> {
    /// The two `Cluster`s that are connected by the spring.
    clusters: [&'a C; 2],
    /// The natural length of the spring.
    l0: T,
    /// The spring constant of the spring.
    k: f32,
    /// The actual length of the spring.
    l: f32,
    /// The magnitude of the force exerted by the spring.
    f_mag: f32,
}

impl<T: Number, C: Cluster<T>> Clone for Spring<'_, T, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Number, C: Cluster<T>> Copy for Spring<'_, T, C> {}

impl<'a, T: Number, C: Cluster<T>> Spring<'a, T, C> {
    /// Create a new `Spring` between two `Cluster`s.
    pub fn new(clusters: [&'a C; 2], k: f32, l0: T, l: f32) -> Self {
        let f_mag = k * (l0.as_f32() - l);
        Self {
            clusters,
            l0,
            k,
            l,
            f_mag,
        }
    }

    /// Deconstruct the `Spring` into its components.
    pub const fn deconstruct(self) -> ([&'a C; 2], f32, T, f32) {
        (self.clusters, self.k, self.l0, self.l)
    }

    /// Get the two `Cluster`s connected by the spring.
    pub const fn clusters(&self) -> [&C; 2] {
        self.clusters
    }

    /// Get the spring constant of the spring.
    pub const fn k(&self) -> f32 {
        self.k
    }

    /// Change the spring constant of the spring.
    pub const fn with_k(mut self, k: f32) -> Self {
        self.k = k;
        self
    }

    /// Get the ratio of the current length to the natural length of the spring.
    pub fn ratio(&self) -> f32 {
        self.l / self.l0.as_f32()
    }

    /// Get the magnitude of the force exerted by the spring.
    pub const fn f_mag(&self) -> f32 {
        self.f_mag
    }

    /// Get the displacement of the spring from its natural length.
    fn dx(&self) -> f32 {
        self.l0.as_f32() - self.l
    }

    /// Update the force exerted by the spring.
    pub fn update_force(&mut self) {
        self.f_mag = self.k * self.dx();
    }

    /// Update the length and force of the spring.
    pub fn update_length(&mut self, l: f32) {
        self.l = l;
        self.update_force();
    }

    /// Get the unit vector of the spring.
    pub fn unit_vector<const DIM: usize, Me>(&self, system: &System<DIM, Me, T, C>) -> Vector<DIM> {
        let [a, b] = self.clusters;
        system[a.arg_center()][0].unit_vector_to(&system[b.arg_center()][0])
    }

    /// Get the potential energy stored in the spring.
    pub fn potential_energy(&self) -> f32 {
        0.5 * self.k * self.dx().powi(2)
    }
}
