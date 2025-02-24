//! A vector that can represent the position, velocity, or force in a mass-
//! spring system.

use rand::prelude::*;

/// A vector that can represent the position, velocity, or force in a mass-
/// spring system.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const DIM: usize>([f32; DIM]);

impl<const DIM: usize> Vector<DIM> {
    /// Create a new `Vector` with the given elements.
    pub const fn new(elements: [f32; DIM]) -> Self {
        Self(elements)
    }

    /// Create a new `Vector` with all elements set to `0.0`.
    pub const fn zero() -> Self {
        Self([0.0; DIM])
    }

    /// Create a new `Vector` with all elements set to `1.0`.
    pub const fn one() -> Self {
        Self([1.0; DIM])
    }

    /// Create a new `Vector` with all elements set to `v`.
    pub const fn fill(v: f32) -> Self {
        Self([v; DIM])
    }

    /// Create a new random `Vector` with elements in the range `[min, max)`.
    pub fn random<R: rand::Rng>(rng: &mut R, min: f32, max: f32) -> Self {
        let mut result = Self::zero();
        for x in result.iter_mut() {
            *x = rng.gen_range(min..max);
        }
        result
    }

    /// Create a new random unit `Vector`.
    pub fn random_unit<R: rand::Rng>(rng: &mut R) -> Self {
        Self::random(rng, -1.0, 1.0).normalized()
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

    /// Get the magnitude of the `Vector`.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        self.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// Get the euclidean distance between two `Vector`s.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f32 {
        distances::simd::euclidean_f32(self.as_slice(), other.as_slice())
    }

    /// Get the unit vector between two `Vector`s.
    pub fn unit_vector_to(&self, other: &Self) -> Self {
        let v = other - self;
        v / v.magnitude()
    }

    /// Normalize the `Vector`.
    pub fn normalized(&self) -> Self {
        *self / self.magnitude()
    }

    /// Normalize the `Vector`.
    pub fn normalize(&mut self) {
        *self /= self.magnitude();
    }

    /// Get the dot product of two `Vector`s.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.iter().zip(other.iter()).map(|(&a, &b)| a * b).sum()
    }
}

impl<const DIM: usize> AsRef<[f32; DIM]> for Vector<DIM> {
    fn as_ref(&self) -> &[f32; DIM] {
        &self.0
    }
}

impl<const DIM: usize> AsRef<[f32]> for Vector<DIM> {
    fn as_ref(&self) -> &[f32] {
        self.0.as_slice()
    }
}

impl<const DIM: usize> core::ops::Deref for Vector<DIM> {
    type Target = [f32; DIM];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const DIM: usize> core::ops::DerefMut for Vector<DIM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Implementations of `+` and `+=` on `Vector`s.
impl<const DIM: usize> core::ops::Add<Self> for Vector<DIM> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] + other[i];
        }
        Self(result)
    }
}

impl<const DIM: usize> core::ops::Add<Self> for &Vector<DIM> {
    type Output = Vector<DIM>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] + rhs[i];
        }
        Vector(result)
    }
}

impl<const DIM: usize> core::ops::AddAssign<Self> for Vector<DIM> {
    fn add_assign(&mut self, other: Self) {
        for i in 0..DIM {
            self[i] += other[i];
        }
    }
}

impl<const DIM: usize> core::ops::AddAssign<Self> for &mut Vector<DIM> {
    fn add_assign(&mut self, other: Self) {
        for i in 0..DIM {
            self[i] += other[i];
        }
    }
}

// Implementations of `-` and `-=` on `Vector`s.
impl<const DIM: usize> core::ops::Sub<Self> for Vector<DIM> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] - other[i];
        }
        Self(result)
    }
}

impl<const DIM: usize> core::ops::Sub<Self> for &Vector<DIM> {
    type Output = Vector<DIM>;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] - other[i];
        }
        Vector(result)
    }
}

impl<const DIM: usize> core::ops::SubAssign<Self> for Vector<DIM> {
    fn sub_assign(&mut self, other: Self) {
        for i in 0..DIM {
            self[i] -= other[i];
        }
    }
}

impl<const DIM: usize> core::ops::Neg for Vector<DIM> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = -self[i];
        }
        Self(result)
    }
}

// Implementations of `*` and `*=` on `Vector`s.
impl<const DIM: usize> core::ops::Mul<f32> for Vector<DIM> {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] * scalar;
        }
        Self(result)
    }
}

impl<const DIM: usize> core::ops::Mul<f32> for &Vector<DIM> {
    type Output = Vector<DIM>;

    fn mul(self, scalar: f32) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] * scalar;
        }
        Vector(result)
    }
}

impl<const DIM: usize> core::ops::MulAssign<f32> for Vector<DIM> {
    fn mul_assign(&mut self, scalar: f32) {
        for i in 0..DIM {
            self[i] *= scalar;
        }
    }
}

impl<const DIM: usize> core::ops::MulAssign<f32> for &mut Vector<DIM> {
    fn mul_assign(&mut self, scalar: f32) {
        for i in 0..DIM {
            self[i] *= scalar;
        }
    }
}

// Implementations of `/` and `/=` on `Vector`s.
impl<const DIM: usize> core::ops::Div<f32> for Vector<DIM> {
    type Output = Self;

    fn div(self, scalar: f32) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] / scalar;
        }
        Self(result)
    }
}

impl<const DIM: usize> core::ops::Div<f32> for &Vector<DIM> {
    type Output = Vector<DIM>;

    fn div(self, scalar: f32) -> Self::Output {
        let mut result = [0.0; DIM];
        for i in 0..DIM {
            result[i] = self[i] / scalar;
        }
        Vector(result)
    }
}

impl<const DIM: usize> core::ops::DivAssign<f32> for Vector<DIM> {
    fn div_assign(&mut self, scalar: f32) {
        for i in 0..DIM {
            self[i] /= scalar;
        }
    }
}

impl<const DIM: usize> core::ops::DivAssign<f32> for &mut Vector<DIM> {
    fn div_assign(&mut self, scalar: f32) {
        for i in 0..DIM {
            self[i] /= scalar;
        }
    }
}
