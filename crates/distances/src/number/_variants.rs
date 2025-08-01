//! Number variants for floats, integers, and unsigned integers.

use core::{hash::Hash, ops::Neg};

use num_integer::Integer;

use crate::Number;

/// Sub-trait of `Number` for all integer types.
pub trait Int: Number + Hash + Eq + Ord {
    /// Returns the Greatest Common Divisor of two integers.
    #[must_use]
    fn gcd(&self, other: &Self) -> Self;

    /// Returns the Least Common Multiple of two integers.
    #[must_use]
    fn lcm(&self, other: &Self) -> Self;
}

/// Macro to implement `IntNumber` for all integer types.
macro_rules! impl_int {
    ($($ty:ty),*) => {
        $(
            impl Int for $ty {
                fn gcd(&self, other: &Self) -> Self {
                    Integer::gcd(&self, other)
                }

                fn lcm(&self, other: &Self) -> Self {
                    Integer::lcm(&self, other)
                }
            }
        )*
    }
}

impl_int!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);

/// Sub-trait of `Number` for all signed integer types.
pub trait IInt: Number + Neg<Output = Self> + Hash + Eq + Ord {}

/// Macro to implement `IIntNumber` for all signed integer types.
macro_rules! impl_iint {
    ($($ty:ty),*) => {
        $(
            impl IInt for $ty {}
        )*
    }
}

impl_iint!(i8, i16, i32, i64, i128, isize);

/// Sub-trait of `Number` for all unsigned integer types.
pub trait UInt: Number + Hash + Eq + Ord {
    /// Returns the number as a `i64`.
    fn as_i64(self) -> i64;
}

/// Macro to implement `UIntNumber` for all unsigned integer types.
macro_rules! impl_uint {
    ($($ty:ty),*) => {
        $(
            #[allow(clippy::cast_lossless, clippy::cast_possible_truncation)]
            impl UInt for $ty {
                #[allow(clippy::cast_possible_wrap)]
                fn as_i64(self) -> i64 {
                    self as i64
                }
            }
        )*
    }
}

impl_uint!(u8, u16, u32, u64, u128, usize);

/// Sub-trait of `Number` for all floating point types.
pub trait Float: Number + core::ops::Neg<Output = Self> {
    /// The square-root of 2.
    const SQRT_2: Self;

    /// The value of e.
    const E: Self;

    /// The value of π.
    const PI: Self;

    /// Returns the reciprocal of `self`.
    #[must_use]
    fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Returns the sign of `self`.
    #[must_use]
    fn signum(self) -> Self;

    /// Returns the square root of a `Float`.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Returns the cube root of a `Float`.
    #[must_use]
    fn cbrt(self) -> Self;

    /// Returns the inverse square root of a `Float`, i.e. `1.0 / self.sqrt()`.
    #[must_use]
    fn inv_sqrt(self) -> Self {
        Self::ONE / self.sqrt()
    }

    /// Returns `self` raised to the power of `exp`.
    #[must_use]
    fn powf(self, exp: Self) -> Self;

    /// The error function.
    #[must_use]
    fn erf(self) -> Self;

    /// Returns the logarithm of `self` with base 2.
    #[must_use]
    fn log2(self) -> Self;

    /// Returns the natural logarithm of `self`.
    #[must_use]
    fn ln(self) -> Self;

    /// Returns the sine of `self`.
    #[must_use]
    fn sin(self) -> Self;

    /// Returns the cosine of `self`.
    #[must_use]
    fn cos(self) -> Self;

    /// Returns the tangent of `self`.
    #[must_use]
    fn tan(self) -> Self {
        self.sin() / self.cos()
    }

    /// Returns the floor of `self`.
    #[must_use]
    fn floor(self) -> Self;

    /// Returns the ceiling of `self`.
    #[must_use]
    fn ceil(self) -> Self;

    /// Returns whether the `Float` is NaN.
    #[must_use]
    fn is_nan(self) -> bool;

    /// Returns whether the `Float` is finite.
    #[must_use]
    fn is_finite(self) -> bool;
}

impl Float for f32 {
    const SQRT_2: Self = core::f32::consts::SQRT_2;

    const E: Self = core::f32::consts::E;

    const PI: Self = core::f32::consts::PI;

    fn signum(self) -> Self {
        self.signum()
    }

    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    fn cbrt(self) -> Self {
        Self::cbrt(self)
    }

    fn powf(self, exp: Self) -> Self {
        Self::powf(self, exp)
    }

    fn erf(self) -> Self {
        libm::erff(self)
    }

    fn log2(self) -> Self {
        Self::log2(self)
    }

    fn ln(self) -> Self {
        Self::ln(self)
    }

    fn sin(self) -> Self {
        Self::sin(self)
    }

    fn cos(self) -> Self {
        Self::cos(self)
    }

    fn floor(self) -> Self {
        Self::floor(self)
    }

    fn ceil(self) -> Self {
        Self::ceil(self)
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

impl Float for f64 {
    const SQRT_2: Self = core::f64::consts::SQRT_2;

    const E: Self = core::f64::consts::E;

    const PI: Self = core::f64::consts::PI;

    fn signum(self) -> Self {
        self.signum()
    }

    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    fn cbrt(self) -> Self {
        Self::cbrt(self)
    }

    fn powf(self, exp: Self) -> Self {
        Self::powf(self, exp)
    }

    fn erf(self) -> Self {
        libm::erf(self)
    }

    fn log2(self) -> Self {
        Self::log2(self)
    }

    fn ln(self) -> Self {
        Self::ln(self)
    }

    fn sin(self) -> Self {
        Self::sin(self)
    }

    fn cos(self) -> Self {
        Self::cos(self)
    }

    fn floor(self) -> Self {
        Self::floor(self)
    }

    fn ceil(self) -> Self {
        Self::ceil(self)
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}
