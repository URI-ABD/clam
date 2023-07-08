//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Number`s.
//! Distance values are also represented as `Number`s.

use core::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};

/// Collections of `Number`s can be used to calculate distances.
pub trait Number:
    Add<Output = Self>
    + AddAssign<Self>
    + Sum<Self>
    + Sub<Output = Self>
    + SubAssign<Self>
    + Mul<Output = Self>
    + MulAssign<Self>
    + Div<Output = Self>
    + DivAssign<Self>
    + Rem<Output = Self>
    + RemAssign<Self>
    + Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + Debug
    + Display
    + Default
{
    /// Returns the additive identity.
    fn zero() -> Self;

    /// Returns the multiplicative identity.
    fn one() -> Self;

    /// Returns `self + a * b`.
    #[must_use]
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Replaces `self` with `self + a * b`.
    fn mul_add_assign(&mut self, a: Self, b: Self);

    /// Casts a number to `Self`. This may be a lossy conversion.
    fn from<T: Number>(n: T) -> Self;

    /// Returns the number as a `f32`. This may be a lossy conversion.
    fn as_f32(self) -> f32;

    /// Returns the number as a `f64`. This may be a lossy conversion.
    fn as_f64(self) -> f64;

    /// Returns the number as a `u64`. This may be a lossy conversion.
    fn as_u64(self) -> u64;

    /// Returns the number as a `i64`. This may be a lossy conversion.
    fn as_i64(self) -> i64;

    /// Returns the absolute value of a `Number`.
    #[must_use]
    fn abs(self) -> Self;

    /// Returns the absolute difference between two `Number`s.
    #[must_use]
    fn abs_diff(self, other: Self) -> Self;

    /// Returns `self` raised to the power of `exp`.
    #[must_use]
    fn powi(self, exp: i32) -> Self;
}

impl Number for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        // libm::fmaf(self, a, b)  // no-std
        self.mul_add(a, b)
    }

    fn mul_add_assign(&mut self, a: Self, b: Self) {
        *self = self.mul_add(a, b);
    }

    fn as_f32(self) -> f32 {
        self
    }

    fn from<T: Number>(n: T) -> Self {
        n.as_f32()
    }

    #[allow(clippy::cast_lossless)]
    fn as_f64(self) -> f64 {
        self as f64
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn as_u64(self) -> u64 {
        self as u64
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_i64(self) -> i64 {
        self as i64
    }

    fn abs(self) -> Self {
        // libm::fabsf(self)  // no-std
        self.abs()
    }

    fn abs_diff(self, other: Self) -> Self {
        (self - other).abs()
    }

    fn powi(self, exp: i32) -> Self {
        // libm::powf(self, exp.as_f32())  // no-std
        self.powi(exp)
    }
}

impl Number for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        // libm::fma(self, a, b)  // no-std
        self.mul_add(a, b)
    }

    fn mul_add_assign(&mut self, a: Self, b: Self) {
        *self = self.mul_add(a, b);
    }

    fn from<T: Number>(n: T) -> Self {
        n.as_f64()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_f32(self) -> f32 {
        self as f32
    }

    fn as_f64(self) -> f64 {
        self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn as_u64(self) -> u64 {
        self as u64
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_i64(self) -> i64 {
        self as i64
    }

    fn abs(self) -> Self {
        // libm::fabs(self)  // no-std
        self.abs()
    }

    fn abs_diff(self, other: Self) -> Self {
        (self - other).abs()
    }

    fn powi(self, exp: i32) -> Self {
        // libm::pow(self, exp.as_f64())  // no-std
        self.powi(exp)
    }
}

/// A macro to implement the `Number` trait for primitive types.
macro_rules! impl_number_iint {
    ($($ty:ty),*) => {
        $(
            impl Number for $ty {
                fn zero() -> Self {
                    0
                }

                fn one() -> Self {
                    1
                }

                fn mul_add(self, a: Self, b: Self) -> Self {
                    self + a * b
                }

                fn mul_add_assign(&mut self, a: Self, b: Self) {
                    *self += a * b;
                }

                fn from<T: Number>(n: T) -> Self {
                    n.as_i64() as $ty
                }

                fn as_f32(self) -> f32 {
                    self as f32
                }

                fn as_f64(self) -> f64 {
                    self as f64
                }

                fn as_u64(self) -> u64 {
                    self as u64
                }

                fn as_i64(self) -> i64 {
                    self as i64
                }

                fn abs(self) -> Self {
                    <$ty>::abs(self)
                }

                fn abs_diff(self, other: Self) -> Self {
                    <$ty>::abs(self - other)
                }

                fn powi(self, exp: i32) -> Self {
                    <$ty>::pow(self, exp as u32)
                }
            }
        )*
    }
}

impl_number_iint!(i8, i16, i32, i64, i128, isize);

/// A macro to implement the `Number` trait for primitive types.
macro_rules! impl_number_uint {
    ($($ty:ty),*) => {
        $(
            impl Number for $ty {
                fn zero() -> Self {
                    0
                }

                fn one() -> Self {
                    1
                }

                fn mul_add(self, a: Self, b: Self) -> Self {
                    self + a * b
                }

                fn mul_add_assign(&mut self, a: Self, b: Self) {
                    *self += a * b;
                }

                fn from<T: Number>(n: T) -> Self {
                    n.as_u64() as $ty
                }

                fn as_f32(self) -> f32 {
                    self as f32
                }

                fn as_f64(self) -> f64 {
                    self as f64
                }

                fn as_u64(self) -> u64 {
                    self as u64
                }

                fn as_i64(self) -> i64 {
                    self as i64
                }

                fn abs(self) -> Self {
                    self
                }

                fn abs_diff(self, other: Self) -> Self {
                    self.abs_diff(other)
                }

                fn powi(self, exp: i32) -> Self {
                    <$ty>::pow(self, exp as u32)
                }
            }
        )*
    }
}

impl_number_uint!(u8, u16, u32, u64, u128, usize);
