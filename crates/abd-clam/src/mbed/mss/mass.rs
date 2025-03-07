//! A `Mass` represents the location of a `Cluster` in the dimension reduction.

use core::marker::PhantomData;

use distances::{number::Multiplication, Number};

use crate::{Cluster, Dataset, Metric};

use super::super::Vector;

/// A `Mass` represents the location of a `Cluster` in the dimension reduction.
#[derive(PartialEq, Clone)]
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

impl<T: Number, C: Cluster<T>, const DIM: usize> core::fmt::Debug for Mass<'_, T, C, DIM> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mass")
            .field("c", &self.arg_center())
            .field("m", &self.m)
            .field("x", &self.x)
            .field("v", &self.v)
            .field("f", &self.f)
            .field("f_mag", &self.f_mag)
            .field("ke", &self.ke)
            .finish()
    }
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

    /// Returns the kinetic energy of the `Mass`.
    pub const fn ke(&self) -> f32 {
        self.ke
    }

    /// Returns the sum of the magnitudes of all forces acting on the `Mass`.
    pub const fn f_mag(&self) -> f32 {
        self.f_mag
    }

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

    /// Returns two new `Mass`es representing the children of the `Cluster` that
    /// this `Mass` represents.
    ///
    /// The child masses will have been moved to form a triangle with the parent
    /// mass.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the original items.
    /// - `metric`: The metric used to compute the distances between the items.
    /// - `x`: The first unit vector along which to move the child masses.
    /// - `y`: The second unit vector along which to move the second child mass.
    ///
    /// # Panics
    ///
    /// - If the `Cluster` that this `Mass` represents does not have exactly two
    ///   children.
    #[allow(clippy::panic, clippy::similar_names)]
    pub(crate) fn child_triangle<I, D: Dataset<I>, M: Metric<I, T>>(
        &self,
        data: &D,
        metric: &M,
        x: Vector<DIM>,
        y: Vector<DIM>,
        scale: f32,
    ) -> [Self; 2] {
        let children = self.c.children();
        if children.len() != 2 {
            let msg = "The cluster does not have exactly two children.";
            ftlog::error!("{msg}");
            panic!("{msg}");
        }
        let (a, b) = (children[0], children[1]);

        let (ac, bc, ab) = (
            data.one_to_one(a.arg_center(), self.arg_center(), metric),
            data.one_to_one(b.arg_center(), self.arg_center(), metric),
            data.one_to_one(a.arg_center(), b.arg_center(), metric),
        );

        let (dxa, dxb, dyb) = triangle_displacements(ac, bc, ab, scale);
        let ax = self.x + x * dxa;
        let bx = self.x + x * dxb + y * dyb;

        [Self::new(a, ax, self.v), Self::new(b, bx, self.v)]
    }
}

/// Compute the displacements of the child masses from the parent mass. We
/// assume that `c` is the parent mass and `a` and `b` are the children.
#[allow(clippy::similar_names)]
fn triangle_displacements<T: Number>(ac: T, bc: T, ab: T, scale: f32) -> (f32, f32, f32) {
    // Since the positions are stored as `f32` arrays, we cast the distances
    // to `f32` for internal computations.
    let (fac, fbc, fab) = (ac.as_f32() * scale, bc.as_f32() * scale, ab.as_f32() * scale);

    // Compute the deltas by which to move the child masses.

    // Note that `a` will only  be moved along the `x` axis while `b` may be
    // moved along one or both axes.

    // Check if the distances form a triangle.
    if crate::utils::is_triangle(ac, bc, ab) {
        // We will move `a` along only the `x` axis and `b` along both axes.

        // Use the law of cosines to compute the length of the projection of
        // `cb` onto the x axis.
        let dxb = (fab.square() - fac.square() - fbc.square()) / (2.0 * fac);

        // Use the Pythagorean theorem to compute the length of the
        // projection of `cb` onto the y axis.
        let dyb = (fbc.square() - dxb.square()).sqrt();

        (fac, dxb, dyb)
    } else {
        // Check if the distances indicate that the three points are colinear.
        let (dxa, dxb) = if crate::utils::is_colinear(ac, bc, ab) {
            // We will move both `a` and `b` along only the `x` axis. We simply
            // need to determine which of the three is in the middle.

            if ab > ac && ab > bc {
                // `ab` is the longest side so `c` in in the middle.
                (-fac, fbc)
            } else if ac > ab && ac > bc {
                // `ac` is the longest side so `b` is in the middle.
                (fac + fab, fbc)
            } else {
                // `bc` is the longest side so `a` is in the middle.
                (fac, fbc + fab)
            }
        } else {
            // The triangle inequality is not satisfied.

            // The only way that the three distances do not form a triangle
            // is if one of them is larger than the sum of the other two.
            // In this case, we will preserve the two shorter distances and
            // ensure that the largest delta is the sum of the two smaller
            // deltas.

            // Note in the following branches that the longest side does not
            // show up in the returned deltas. This is the only difference
            // between this case and the colinear case.
            if ab > ac && ab > bc {
                // `ab` is the longest side so `c` in in the middle.
                (-fac, fbc)
            } else if ac > ab && ac > bc {
                // `ac` is the longest side so `b` is in the middle.
                (fbc + fab, fbc)
            } else {
                // `bc` is the longest side so `a` is in the middle.
                (fac, fac + fab)
            }
        };

        (dxa, dxb, 0.0)
    }
}
