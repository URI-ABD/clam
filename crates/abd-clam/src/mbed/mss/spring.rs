//! A `Spring` connecting two `Mass`es in a mass-spring system.

use distances::{number::Float, Number};
use slotmap::{DefaultKey, HopSlotMap};

use crate::Cluster;

use super::Mass;

use super::super::Vector;

/// A spring connecting two `Mass`es in a mass-spring system.
#[derive(PartialEq, Eq, Clone)]
#[must_use]
pub struct Spring<F: Float, const DIM: usize> {
    /// The first `Mass` connected by the spring.
    a: DefaultKey,
    /// The second `Mass` connected by the spring.
    b: DefaultKey,
    /// The spring constant of the spring.
    k: F,
    /// The natural length of the spring.
    l0: F,
    /// The actual length of the spring.
    l: F,
    /// The displacement of the spring from its natural length.
    dx: F,
    /// The displacement ratio of the spring from its natural length.
    ratio: F,
    /// The potential energy stored in the spring.
    pe: F,
    /// The force exerted by the spring directed from `a` to `b`.
    f: Vector<F, DIM>,
    /// The number of times the spring has been loosened.
    times_loosened: usize,
    /// Whether the spring connects two leaf masses.
    connects_leaves: bool,
    /// The order of the force function.
    n: i32,
}

impl<F: Float, const DIM: usize> core::fmt::Debug for Spring<F, DIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spring")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("k", &self.k)
            .field("l0", &self.l0)
            .field("l", &self.l)
            .field("dx", &self.dx)
            .field("ratio", &self.ratio)
            .field("pe", &self.pe)
            .field("f", &self.f)
            .field("times_loosened", &self.times_loosened)
            .field("connects_leaves", &self.connects_leaves)
            .field("order", &self.n)
            .finish()
    }
}

impl<F: Float, const DIM: usize> Spring<F, DIM> {
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
        a: DefaultKey,
        b: DefaultKey,
        masses: &HopSlotMap<DefaultKey, Mass<T, C, F, DIM>>,
        k: F,
        l0: F,
        times_loosened: usize,
        connects_leaves: bool,
    ) -> Self {
        // let [ax, bx] = [masses[a].x(), masses[b].x()];

        let mut s = Self {
            a,
            b,
            k,
            l0,
            l: F::ZERO,
            dx: F::ZERO,
            ratio: F::ZERO,
            pe: F::ZERO,
            f: Vector::zero(),
            times_loosened,
            connects_leaves,
            n: 2,
        };
        s.recalculate(masses);

        s
    }

    /// Returns whether the `arg_center` of `a` is less than the `arg_center` of
    /// `b`.
    pub fn is_ordered<T: Number, C: Cluster<T>>(&self, masses: &HopSlotMap<DefaultKey, Mass<T, C, F, DIM>>) -> bool {
        masses[self.a].arg_center() < masses[self.b].arg_center()
    }

    /// Returns whether the `arg_center` of `a` is same as the `arg_center` of
    /// `b`.
    pub fn is_circular<T: Number, C: Cluster<T>>(&self, masses: &HopSlotMap<DefaultKey, Mass<T, C, F, DIM>>) -> bool {
        masses[self.a].arg_center() == masses[self.b].arg_center()
    }

    /// Loosens the spring by a multiplicative `factor`.
    pub fn loosen(&mut self, factor: F) {
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
    pub const fn k(&self) -> F {
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
    pub fn recalculate<T: Number, C: Cluster<T>>(&mut self, masses: &HopSlotMap<DefaultKey, Mass<T, C, F, DIM>>) {
        let [ax, bx] = [masses[self.a].x(), masses[self.b].x()];

        self.l = ax.distance_to(bx);
        self.dx = self.l0 - self.l;
        self.ratio = self.dx.abs() / self.l0;
        self.pe = pe(self.k, self.l0, self.l, self.n);

        let f_mag = f_mag(self.k, self.l0, self.l, self.n);
        let uv = ax.unit_vector_to(bx);
        self.f = uv * f_mag;
    }

    /// Returns the `Index`es of the `Mass`es connected by the spring.
    pub const fn mass_indices(&self) -> [DefaultKey; 2] {
        [self.a, self.b]
    }

    /// Returns the force exerted by the spring. The force is directed from `a`
    /// to `b`.
    pub const fn f(&self) -> &Vector<F, DIM> {
        &self.f
    }

    /// Returns the potential energy stored in the spring.
    pub const fn pe(&self) -> F {
        self.pe
    }

    /// Returns the displacement of the spring from its natural length.
    pub const fn dx(&self) -> F {
        self.dx
    }

    /// Returns the displacement ratio of the spring from its natural length.
    pub const fn ratio(&self) -> F {
        self.ratio
    }

    /// Returns whether the spring connects the `Mass` with the given `Index`.
    pub fn connects(&self, i: DefaultKey) -> bool {
        i == self.a || i == self.b
    }

    /// Returns the neighbor of the `Mass` connected to the given `Index` if it
    /// is connected by the spring, otherwise returns `None`.
    pub fn neighbor_of(&self, i: DefaultKey) -> Option<DefaultKey> {
        if i == self.a {
            Some(self.b)
        } else if i == self.b {
            Some(self.a)
        } else {
            None
        }
    }
}

/// Returns the magnitude of the force exerted by the spring.
fn f_mag<F: Float>(k: F, l0: F, l: F, n: i32) -> F {
    k * (l.sqrt() - l.recip().powi(n) - l0.sqrt() + l0.recip().powi(n))
}

/// Returns the potential energy stored in the spring.
fn pe<F: Float>(k: F, l0: F, l: F, n: i32) -> F {
    let pe = |x: F| {
        x * l0.recip().powi(n) - x * l0.sqrt() + x.powi(1 - n) / F::from(n - 1) + x.sqrt().cube().double() / F::from(3)
    };
    k * (pe(l) - pe(l0))
}
