//! A `Spring` between two `Cluster`s in a mass-spring system.

use distances::{
    number::{Float, Multiplication},
    Number,
};

use crate::Cluster;

use super::{MassSpringSystem, Vector};

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
    /// The displacement of the spring from its natural length.
    dx: f32,
    /// The magnitude of the force exerted by the spring.
    f_mag: f32,
    /// The potential energy stored in the spring.
    pe: f32,
    /// The number of times the spring has been loosened.
    num_loosened: usize,
}

impl<T: Number, C: Cluster<T>> Clone for Spring<'_, T, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Number, C: Cluster<T>> Copy for Spring<'_, T, C> {}

impl<'a, T: Number, C: Cluster<T>> Spring<'a, T, C> {
    /// Create a new `Spring` between two `Cluster`s.
    pub fn new(clusters: [&'a C; 2], l0: T, l: f32, num_loosened: usize, dk: f32) -> Self {
        let k = l0.as_f32();
        let dx = l - l0.as_f32();
        let k = k * dk.powi(num_loosened.as_i32());
        let f_mag = f_mag(k, l0.as_f32(), l, 2);
        let pe = pe(k, l0.as_f32(), l, 2);
        Self {
            clusters,
            l0,
            k,
            l,
            dx,
            f_mag,
            pe,
            num_loosened,
        }
    }

    /// Deconstruct the `Spring` into its components.
    pub const fn deconstruct(self) -> ([&'a C; 2], f32, T, f32, usize) {
        (self.clusters, self.k, self.l0, self.l, self.num_loosened)
    }

    /// Get the two `Cluster`s connected by the spring.
    pub const fn clusters(&self) -> [&C; 2] {
        self.clusters
    }

    /// Get the spring constant of the spring.
    pub const fn k(&self) -> f32 {
        self.k
    }

    /// Get the displacement of the spring from its natural length.
    pub const fn dx(&self) -> f32 {
        self.dx
    }

    /// Loosen the spring by a factor of `factor`.
    pub fn loosen(&mut self, factor: f32) {
        self.num_loosened += 1;
        self.k *= factor;
        self.f_mag = -self.k * self.dx;
        self.pe = 0.5 * self.k * self.dx.square();
    }

    /// Return whether the spring has been yet to be loosened too many times.
    pub const fn is_intact(&self, threshold: usize) -> bool {
        self.num_loosened < threshold
    }

    /// Get the magnitude of the force exerted by the spring.
    pub const fn f_mag(&self) -> f32 {
        self.f_mag
    }

    /// Update the length of the spring, and recalculate subsequent properties.
    pub fn update_length(&mut self, l: f32) {
        self.dx += l - self.l;
        self.l = l;
        self.f_mag = f_mag(self.k, self.l0.as_f32(), self.l, 2);
        self.pe = pe(self.k, self.l0.as_f32(), self.l, 2);
    }

    /// Get the unit vector of the spring.
    pub fn unit_vector<const DIM: usize, Me>(&self, system: &MassSpringSystem<DIM, Me, T, C>) -> Vector<f32, DIM> {
        let [a, b] = self.clusters;
        system[a.arg_center()][0].unit_vector_to(&system[b.arg_center()][0])
    }

    /// Get the potential energy stored in the spring.
    pub const fn potential_energy(&self) -> f32 {
        self.pe
    }
}

/// Returns the magnitude of the force exerted by the spring.
fn f_mag<F: Float>(k: F, l0: F, l: F, n: i32) -> F {
    k * (l.sqrt() - l.recip().powi(n) - l0.sqrt() + l0.recip().powi(n))
}

/// Returns the potential energy stored in the spring.
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
