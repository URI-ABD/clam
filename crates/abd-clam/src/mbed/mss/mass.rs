//! A `Mass` represents the location of a `Cluster` in the dimension reduction.

use core::marker::PhantomData;

use distances::{number::Multiplication, Number};

use crate::Cluster;

use super::super::Vector;

/// A `Mass` represents the location of a `Cluster` in the dimension reduction.
#[derive(Debug, PartialEq, Clone)]
#[must_use]
pub struct Mass<'a, T: Number, C: Cluster<T>, const DIM: usize> {
    /// The index of the center of the cluster that this `Mass` represents.
    c: &'a C,
    /// The mass, usually equal to the cardinality, of the cluster.
    m: f32,
    /// The position of the `Mass`.
    x: Vector<DIM>,
    /// The velocity of the `Mass`.
    v: Vector<DIM>,
    /// The force acting on the `Mass`.
    f: Vector<DIM>,
    /// The sum of the magnitudes of all forces acting on the `Mass`.
    f_mag: f32,
    /// The kinetic energy of the `Mass`.
    ke: f32,
    /// Satisfying the compiler.
    phantom: PhantomData<T>,
}

impl<'a, T: Number, C: Cluster<T>, const DIM: usize> Mass<'a, T, C, DIM> {
    /// Create a new `Mass` representing a `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `c`: The `Cluster` that the `Mass` represents.
    /// - `x`: The initial position of the `Mass`.
    /// - `v`: The initial velocity of the `Mass`.
    #[allow(clippy::many_single_char_names)]
    pub fn new(c: &'a C, x: Vector<DIM>, v: Vector<DIM>) -> Self {
        let m = c.cardinality().as_f32();
        let f = Vector::zero();
        let ke = m.half() * v.magnitude().square();

        Self {
            c,
            m,
            x,
            v,
            f,
            f_mag: 0.0,
            ke,
            phantom: PhantomData,
        }
    }

    /// Returns the index if the center of the `Cluster` that this `Mass`
    /// represents.
    pub fn arg_center(&self) -> usize {
        self.c.arg_center()
    }

    /// Returns the indices of the `Cluster` that this `Mass` represents.
    pub fn indices(&self) -> Vec<usize> {
        self.c.indices()
    }

    /// Returns whether the `Cluster` that this `Mass` represents is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.c.is_leaf()
    }

    /// Returns the position of the `Mass`.
    pub const fn x(&self) -> &Vector<DIM> {
        &self.x
    }

    // /// Returns the velocity of the `Mass`.
    // pub const fn v(&self) -> &Vector<DIM> {
    //     &self.v
    // }

    /// Returns the kinetic energy of the `Mass`.
    pub const fn ke(&self) -> f32 {
        self.ke
    }

    // /// Returns the force acting on the `Mass`.
    // pub const fn f(&self) -> &Vector<DIM> {
    //     &self.f
    // }

    // /// Changes the position of the `Mass`.
    // pub const fn with_x(mut self, x: &Vector<DIM>) -> Self {
    //     self.x = *x;
    //     self
    // }

    // /// Changes the velocity of the `Mass`.
    // pub fn with_v(mut self, v: &Vector<DIM>) -> Self {
    //     self.v = *v;
    //     self.ke = self.m.half() * v.magnitude().square();
    //     self
    // }

    /// Adds to the force acting on the `Mass`.
    pub fn add_f(&mut self, f: &Vector<DIM>) {
        self.f += *f;
        self.f_mag += f.magnitude();
    }

    /// Applies the force acting on the `Mass`, moving it, and resetting the
    /// force acting on it.
    pub fn apply_f(&mut self, dt: f32, drag: f32) {
        let a = (self.f - self.v * drag) / self.m;
        self.v += a * dt;
        self.x += self.v * dt;

        self.ke = self.m.half() * self.v.magnitude().square();
        self.f = Vector::zero();
        self.f_mag = 0.0;
    }
}
