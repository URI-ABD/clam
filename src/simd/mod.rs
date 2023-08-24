//! Provides simd-accelerated euclidean distance functions for vectors.
#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::must_use_candidate
)]

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
    Vectorized::distance(a, b)
}

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    Vectorized::squared_distance(a, b)
}

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_f64(a: &[f64], b: &[f64]) -> f64 {
    Vectorized::distance(a, b)
}

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_sq_f64(a: &[f64], b: &[f64]) -> f64 {
    Vectorized::squared_distance(a, b)
}

#[macro_use]
mod macros;

mod f32x4;
mod f32x8;

mod f64x2;
mod f64x4;

pub use f32x4::F32x4;
pub use f32x8::F32x8;
pub use f64x2::F64x2;
pub use f64x4::F64x4;

pub(crate) trait Naive {
    type Output;
    type Ty;

    fn squared_distance(self, other: Self) -> Self::Output;
    fn distance(self, other: Self) -> Self::Output;
}

pub(crate) trait Vectorized {
    type Output;
    fn squared_distance(self, other: Self) -> Self::Output;
    fn distance(self, other: Self) -> Self::Output;
}

impl_naive!(f64, f64);
impl_naive!(f32, f32);

/// Calculate the euclidean distance between two slices of equal length
///
/// # Panics
///
/// Will panic if the lengths of the slices are not equal
#[allow(dead_code)]
pub(crate) fn scalar_euclidean<T: Naive>(a: T, b: T) -> T::Output {
    Naive::distance(a, b)
}

/// SIMD-capable calculation of the euclidean distance between two slices
/// of equal length
///
/// # Panics
///
/// Will panic if the lengths of the slices are not equal
#[allow(dead_code)]
pub(crate) fn vector_euclidean<T: Vectorized>(a: T, b: T) -> T::Output {
    Vectorized::distance(a, b)
}

impl Vectorized for &[f32] {
    type Output = f32;
    fn squared_distance(self, other: Self) -> Self::Output {
        if self.len() >= 64 {
            F32x8::squared_distance(self, other)
        } else {
            F32x4::squared_distance(self, other)
        }
    }

    fn distance(self, other: Self) -> Self::Output {
        Vectorized::squared_distance(self, other).sqrt()
    }
}

impl Vectorized for &Vec<f32> {
    type Output = f32;
    fn squared_distance(self, other: Self) -> Self::Output {
        if self.len() >= 64 {
            F32x8::squared_distance(self, other)
        } else {
            F32x4::squared_distance(self, other)
        }
    }

    fn distance(self, other: Self) -> Self::Output {
        Vectorized::squared_distance(self, other).sqrt()
    }
}

impl Vectorized for &[f64] {
    type Output = f64;
    fn squared_distance(self, other: Self) -> Self::Output {
        if self.len() >= 16 {
            F64x4::squared_distance(self, other)
        } else {
            F64x2::squared_distance(self, other)
        }
    }

    fn distance(self, other: Self) -> Self::Output {
        Vectorized::squared_distance(self, other).sqrt()
    }
}

impl Vectorized for &Vec<f64> {
    type Output = f64;
    fn squared_distance(self, other: Self) -> Self::Output {
        if self.len() >= 16 {
            F64x4::squared_distance(self, other)
        } else {
            F64x2::squared_distance(self, other)
        }
    }

    fn distance(self, other: Self) -> Self::Output {
        Vectorized::squared_distance(self, other).sqrt()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    pub const XS: [f32; 72] = [
        6.1125, 10.795, 20.0, 0.0, 10.55, 10.63, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.26, 10.73, 0.0,
        0.0, 20.0, 0.0, 10.4975, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 20.0, 20.0,
        0.0, 0.0, 0.0, 0.0, 10.475, 6.0905, 20.0, 0.0, 20.0, 20.0, 0.0, 10.5375, 10.54, 10.575,
        0.0, 0.0, 0.0, 10.76, 10.755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0,
        20.0, 0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0,
    ];
    pub const YS: [f32; 72] = [
        6.0905, 20.0, 0.0, 20.0, 20.0, 0.0, 10.5375, 10.54, 10.575, 0.0, 0.0, 0.0, 10.76, 10.755,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 20.0, 0.0, 0.0,
        20.0, 0.0, 0.0, 0.0, 20.0, 6.1125, 10.795, 20.0, 0.0, 10.55, 10.63, 20.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 10.26, 10.73, 0.0, 0.0, 20.0, 0.0, 10.4975, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        20.0, 0.0, 20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 10.475,
    ];

    #[test]
    fn verify() {
        for i in 0..XS.len() {
            let x = &XS[..i];
            let y = &YS[..i];
            let res = scalar_euclidean(x, y);
            assert!(
                (Vectorized::distance(x, y) - res).abs() < 0.0001,
                "iter {}, {} != {}",
                i,
                Vectorized::distance(x, y),
                res
            );
            assert!(
                (F32x8::distance(x, y) - res).abs() < 0.0001,
                "iter {}, {} != {}",
                i,
                F32x8::distance(x, y),
                res
            );
            assert!(
                (F32x4::distance(x, y) - res).abs() < 0.0001,
                "iter {}, {} != {}",
                i,
                F32x4::distance(x, y),
                res
            );
        }
    }

    #[test]
    fn verify_random() {
        use symagen::random_data;

        let input_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

        for i in input_sizes {
            let data = random_data::random_f32(2, i, -10.0, 10.0, 42);
            let (a, b) = (&data[0], &data[1]);

            let diff = (vector_euclidean(a, b) - scalar_euclidean(a, b)).abs();
            assert!(diff <= 1e-4, "diff = {}, len = {}", diff, i);
        }
    }

    #[test]
    fn smoke_mul() {
        let a = F32x4::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::from_slice(&[4.0, 3.0, 2.0, 1.0]);
        let c = a * b;
        assert_eq!(c.horizontal_add(), 4.0 + 6.0 + 6.0 + 4.0);
    }

    #[test]
    fn smoke_mul_assign() {
        let mut a = F32x4::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::from_slice(&[4.0, 3.0, 2.0, 1.0]);
        a *= b;
        assert_eq!(a.horizontal_add(), 4.0 + 6.0 + 6.0 + 4.0);
    }

    #[test]
    fn smoke_add() {
        let a = F32x4::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::from_slice(&[4.0, 3.0, 2.0, 1.0]);
        let c = a + b;
        assert_eq!(c, F32x4::new(5.0, 5.0, 5.0, 5.0));
    }

    #[test]
    fn smoke_sub() {
        let a = F32x4::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x4::from_slice(&[4.0, 3.0, 2.0, 1.0]);
        let c = a - b;
        assert_eq!(c, F32x4::new(-3.0, -1.0, 1.0, 3.0));
    }
}
