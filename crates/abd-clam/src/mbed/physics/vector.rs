//! A vector that can represent the position, velocity, or force in a mass-
//! spring system.

use rand::{distr::uniform::SampleUniform, prelude::*};
use std::iter::Sum;

use crate::FloatDistanceValue;

/// A vector that can represent the position, velocity, or force in a mass-
/// spring system.
#[must_use]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Vector<F: FloatDistanceValue, const DIM: usize>([F; DIM]);

impl<F: FloatDistanceValue + core::fmt::Debug, const DIM: usize> core::fmt::Debug for Vector<F, DIM> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl<F: FloatDistanceValue, const DIM: usize> Vector<F, DIM> {
    /// Create a new `Vector` with the given elements.
    pub const fn new(elements: [F; DIM]) -> Self {
        Self(elements)
    }

    /// Create a new `Vector` with all elements set to `0.0`.
    pub fn zero() -> Self {
        Self([F::zero(); DIM])
    }

    /// Create a new `Vector` with all elements set to `1.0`.
    pub fn one() -> Self {
        Self([F::one(); DIM])
    }

    /// Create a new `Vector` with all elements set to `v`.
    pub const fn fill(v: F) -> Self {
        Self([v; DIM])
    }

    /// Get the magnitude of the `Vector`.
    #[must_use]
    pub fn magnitude(&self) -> F {
        self.distance_to(&Self::zero())
    }

    /// Get the euclidean distance between two `Vector`s.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> F {
        self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(&a, &b)| a - b)
            .map(|d| d * d)
            .sum::<F>()
            .sqrt()
    }

    /// Get the unit vector between two `Vector`s.
    ///
    /// If the two `Vector`s are the same, the result will be `[1.0, 0.0, ...,
    /// 0.0]`.
    pub fn unit_vector_to(&self, other: &Self) -> Self {
        let v = other - self;
        let v_mag = v.magnitude();
        if v_mag == F::zero() {
            let mut v = Self::zero();
            v[0] = F::one();
            v
        } else {
            v / v_mag
        }
    }

    /// Normalize the `Vector`. If the magnitude is `0.0`, the `Vector` will be
    /// set to `[1.0, 0.0, ..., 0.0]`.
    pub fn normalized(&self) -> Self {
        let m = self.magnitude();
        if m == F::zero() {
            let mut v = *self;
            v[0] = F::one();
            v
        } else {
            *self / m
        }
    }

    /// Normalize the `Vector`.
    pub fn normalize(&mut self) {
        *self /= self.magnitude();
    }

    /// Get the dot product of two `Vector`s.
    #[must_use]
    pub fn dot(&self, other: &Self) -> F {
        self.iter().zip(other.iter()).map(|(&a, &b)| a * b).sum()
    }

    /// Return the `Vector` has any NANs.
    #[must_use]
    pub fn has_nan(&self) -> bool {
        self.iter().any(|&x| x.is_nan())
    }

    /// Return the `Vector` has any INFs.
    #[must_use]
    pub fn has_inf(&self) -> bool {
        self.iter().any(|&x| !x.is_finite())
    }
}

impl<F: FloatDistanceValue + SampleUniform, const DIM: usize> Vector<F, DIM> {
    /// Create a new random `Vector` with elements in the range `[min, max)`.
    pub fn random<R: rand::Rng>(rng: &mut R, min: F, max: F) -> Self {
        let mut r_f64s = Self::zero();
        for x in r_f64s.iter_mut() {
            *x = rng.random_range(min..max);
        }
        r_f64s
    }

    /// Create a new random unit `Vector`.
    pub fn random_unit<R: rand::Rng>(rng: &mut R) -> Self {
        Self::random(rng, -F::one(), F::one()).normalized()
    }

    /// Get a new unit `Vector` that is perpendicular to this `Vector`.
    pub fn perpendicular<R: rand::Rng>(&self, rng: &mut R) -> Self {
        let (x, y) = {
            let mut v = (0..DIM).collect::<Vec<_>>();
            v.shuffle(rng);
            (v[0], v[1])
        };

        let mut result = self.normalized();
        result[x] = self[y];
        result[y] = -self[x];

        result
    }
}

impl<F: FloatDistanceValue, const DIM: usize> From<Vector<F, DIM>> for [F; DIM] {
    fn from(val: Vector<F, DIM>) -> [F; DIM] {
        val.0
    }
}

impl<F: FloatDistanceValue, const DIM: usize> From<&Vector<F, DIM>> for [F; DIM] {
    fn from(val: &Vector<F, DIM>) -> [F; DIM] {
        val.0
    }
}

impl<F: FloatDistanceValue, const DIM: usize> AsRef<[F; DIM]> for Vector<F, DIM> {
    fn as_ref(&self) -> &[F; DIM] {
        &self.0
    }
}

impl<F: FloatDistanceValue, const DIM: usize> AsRef<[F]> for Vector<F, DIM> {
    fn as_ref(&self) -> &[F] {
        self.0.as_slice()
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::Deref for Vector<F, DIM> {
    type Target = [F; DIM];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::DerefMut for Vector<F, DIM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Implementations of `+` and `+=` on `Vector`s.
impl<F: FloatDistanceValue, const DIM: usize> core::ops::Add<Self> for Vector<F, DIM> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] + other[i];
        }
        Self(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::Add<Self> for &Vector<F, DIM> {
    type Output = Vector<F, DIM>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] + rhs[i];
        }
        Vector(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::AddAssign<Self> for Vector<F, DIM> {
    fn add_assign(&mut self, other: Self) {
        for i in 0..DIM {
            self[i] += other[i];
        }
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::AddAssign<Self> for &mut Vector<F, DIM> {
    fn add_assign(&mut self, other: Self) {
        for i in 0..DIM {
            self[i] += other[i];
        }
    }
}

// Implementations of `-` and `-=` on `Vector`s.
impl<F: FloatDistanceValue, const DIM: usize> core::ops::Sub<Self> for Vector<F, DIM> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] - other[i];
        }
        Self(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::Sub<Self> for &Vector<F, DIM> {
    type Output = Vector<F, DIM>;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] - other[i];
        }
        Vector(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::SubAssign<Self> for Vector<F, DIM> {
    fn sub_assign(&mut self, other: Self) {
        for i in 0..DIM {
            self[i] -= other[i];
        }
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::Neg for Vector<F, DIM> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = -self[i];
        }
        Self(result)
    }
}

// Implementations of `*` and `*=` on `Vector`s.
impl<F: FloatDistanceValue, const DIM: usize> core::ops::Mul<F> for Vector<F, DIM> {
    type Output = Self;

    fn mul(self, scalar: F) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] * scalar;
        }
        Self(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::Mul<F> for &Vector<F, DIM> {
    type Output = Vector<F, DIM>;

    fn mul(self, scalar: F) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] * scalar;
        }
        Vector(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::MulAssign<F> for Vector<F, DIM> {
    fn mul_assign(&mut self, scalar: F) {
        for i in 0..DIM {
            self[i] *= scalar;
        }
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::MulAssign<F> for &mut Vector<F, DIM> {
    fn mul_assign(&mut self, scalar: F) {
        for i in 0..DIM {
            self[i] *= scalar;
        }
    }
}

// Implementations of `/` and `/=` on `Vector`s.
impl<F: FloatDistanceValue, const DIM: usize> core::ops::Div<F> for Vector<F, DIM> {
    type Output = Self;

    fn div(self, scalar: F) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] / scalar;
        }
        Self(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::Div<F> for &Vector<F, DIM> {
    type Output = Vector<F, DIM>;

    fn div(self, scalar: F) -> Self::Output {
        let mut result = [F::zero(); DIM];
        for i in 0..DIM {
            result[i] = self[i] / scalar;
        }
        Vector(result)
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::DivAssign<F> for Vector<F, DIM> {
    fn div_assign(&mut self, scalar: F) {
        for i in 0..DIM {
            self[i] /= scalar;
        }
    }
}

impl<F: FloatDistanceValue, const DIM: usize> core::ops::DivAssign<F> for &mut Vector<F, DIM> {
    fn div_assign(&mut self, scalar: F) {
        for i in 0..DIM {
            self[i] /= scalar;
        }
    }
}

impl<F: FloatDistanceValue, const DIM: usize> Sum for Vector<F, DIM> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, v| acc + v)
    }
}
