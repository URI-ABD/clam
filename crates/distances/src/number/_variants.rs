//! Number variants for floats, integers, and unsigned integers.

use core::hash::Hash;

use crate::Number;

/// Sub-trait of `Number` for all integer types.
pub trait Int: Number + Hash + Eq + Ord {}

/// Macro to implement `IntNumber` for all integer types.
macro_rules! impl_int {
    ($($ty:ty),*) => {
        $(
            impl Int for $ty {}
        )*
    }
}

impl_int!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);

/// Sub-trait of `Number` for all signed integer types.
pub trait IInt: Number + Hash + Eq + Ord {}

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

    /// Returns the number as a `u64`.
    fn as_u64(self) -> u64;
}

/// Macro to implement `UIntNumber` for all unsigned integer types.
macro_rules! impl_uint {
    ($($ty:ty),*) => {
        $(
            impl UInt for $ty {
                fn as_i64(self) -> i64 {
                    self as i64
                }

                fn as_u64(self) -> u64 {
                    self as u64
                }
            }
        )*
    }
}

impl_uint!(u8, u16, u32, u64, u128, usize);

/// Sub-trait of `Number` for all floating point types.
pub trait Float: Number + core::ops::Neg<Output = Self> {
    /// Returns the square root of a `Float`.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Returns the cube root of a `Float`.
    #[must_use]
    fn cbrt(self) -> Self;

    /// Returns the inverse square root of a `Float`, i.e. `1.0 / self.sqrt()`.
    #[must_use]
    fn inv_sqrt(self) -> Self {
        Self::one() / self.sqrt()
    }

    /// Returns `self` raised to the power of `exp`.
    #[must_use]
    fn powf(self, exp: Self) -> Self;
}

/// Macro to implement `UIntNumber` for all unsigned integer types.
macro_rules! impl_float {
    ($($ty:ty),*) => {
        $(
            impl Float for $ty {
                fn sqrt(self) -> Self {
                    Self::sqrt(self)
                }

                fn cbrt(self) -> Self {
                    Self::cbrt(self)
                }

                fn powf(self, exp: Self) -> Self {
                    Self::powf(self, exp)
                }
            }
        )*
    }
}

impl_float!(f32, f64);
