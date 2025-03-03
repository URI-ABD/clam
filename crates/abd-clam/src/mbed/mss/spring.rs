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
    /// The number of times the spring has been loosened.
    times_loosened: usize,
    /// Whether the spring connects two leaf masses.
    connects_leaves: bool,
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
    /// - `times_loosened`: The number of times the spring has been loosened.
    /// - `is_leaf_spring`: Whether the spring connects two leaf masses.
    #[allow(clippy::many_single_char_names)]
    pub fn new<T: Number, C: Cluster<T>>(
        a: Index,
        b: Index,
        masses: &Arena<Mass<T, C, DIM>>,
        k: f32,
        l0: f32,
        times_loosened: usize,
        connects_leaves: bool,
    ) -> Self {
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
            times_loosened,
            connects_leaves,
        }
    }

    /// Loosens the spring by a multiplicative `factor`.
    pub fn loosen(&mut self, factor: f32) {
        self.ratio *= factor;
        self.pe *= factor;
        self.f *= factor;
        self.k *= factor;
        self.times_loosened += 1;
    }

    /// Returns the number of times the spring has been loosened.
    pub const fn times_loosened(&self) -> usize {
        self.times_loosened
    }

    /// Returns the spring constant of the spring.
    pub const fn k(&self) -> f32 {
        self.k
    }

    /// Returns whether the spring is too loose based on the given `threshold`.
    pub const fn is_too_loose(&self, threshold: usize) -> bool {
        self.times_loosened >= threshold
    }

    /// Returns whether the spring connects two leaf masses.
    pub const fn is_leaf_spring(&self) -> bool {
        self.connects_leaves
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

    /// Returns whether the spring connects the `Mass` with the given `Index`.
    pub fn connects(&self, i: Index) -> bool {
        i == self.a || i == self.b
    }

    /// Returns the neighbor of the `Mass` connected to the given `Index` if it
    /// is connected by the spring, otherwise returns `None`.
    pub fn neighbor_of(&self, i: Index) -> Option<Index> {
        if i == self.a {
            Some(self.b)
        } else if i == self.b {
            Some(self.a)
        } else {
            None
        }
    }
}
