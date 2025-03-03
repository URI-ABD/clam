//! A `Spring` connecting two `Mass`es in a mass-spring system.

use distances::Number;
use generational_arena::{Arena, Index};

use crate::Cluster;

use super::Mass;

use super::super::Vector;

/// A spring connecting two `Mass`es in a mass-spring system.
#[derive(Debug, PartialEq, Clone)]
#[must_use]
pub struct Spring<const DIM: usize> {
    /// The first `Mass` connected by the spring.
    a: Index,
    /// The second `Mass` connected by the spring.
    b: Index,
    /// The spring constant of the spring.
    k: f32,
    /// The natural length of the spring.
    l0: f32,
    /// The actual length of the spring.
    l: f32,
    /// The displacement of the spring from its natural length.
    dx: f32,
    /// The displacement ratio of the spring from its natural length.
    ratio: f32,
    /// The potential energy stored in the spring.
    pe: f32,
    /// The force exerted by the spring directed from `a` to `b`.
    f: Vector<DIM>,
}

impl<const DIM: usize> Spring<DIM> {
    /// Create a new `Spring` between two `Mass`es.
    ///
    /// # Arguments
    ///
    /// - `a`: The `Index` of the first `Mass` connected by the spring.
    /// - `b`: The `Index` of the second `Mass` connected by the spring.
    /// - `masses`: The `Arena` of `Mass`es in the mass-spring system.
    /// - `k`: The spring constant of the spring.
    /// - `l0`: The natural length of the spring.
    #[allow(clippy::many_single_char_names)]
    pub fn new<T: Number, C: Cluster<T>>(a: Index, b: Index, masses: &Arena<Mass<T, C, DIM>>, k: f32, l0: f32) -> Self {
        let [ax, bx] = [&masses[a].x(), &masses[b].x()];

        let l = ax.distance_to(bx);
        let dx = l - l0;
        let ratio = dx.abs() / l0;
        let pe = 0.5 * k * dx.square();

        let f_mag = -k * dx;
        let uv = ax.unit_vector_to(bx);
        let f = uv * f_mag;

        Self {
            a,
            b,
            k,
            l0,
            l,
            dx,
            ratio,
            pe,
            f,
        }
    }

    /// Recalculates the properties of the spring.
    pub fn recalculate<T: Number, C: Cluster<T>>(&mut self, masses: &Arena<Mass<T, C, DIM>>) {
        let [ax, bx] = [&masses[self.a].x(), &masses[self.b].x()];

        self.l = ax.distance_to(bx);
        self.dx = self.l - self.l0;
        self.ratio = self.dx.abs() / self.l0;
        self.pe = 0.5 * self.k * self.dx.square();

        let f_mag = -self.k * self.dx;
        let uv = ax.unit_vector_to(bx);
        self.f = uv * f_mag;
    }

    /// Returns the `Index`es of the `Mass`es connected by the spring.
    pub const fn mass_indices(&self) -> [Index; 2] {
        [self.a, self.b]
    }

    /// Returns the force exerted by the spring. The force is directed from `a`
    /// to `b`.
    pub const fn f(&self) -> &Vector<DIM> {
        &self.f
    }

    /// Returns the potential energy stored in the spring.
    pub const fn pe(&self) -> f32 {
        self.pe
    }
}
