//! Addition and Multiplication of `Number` types.

use core::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};

/// The `Addition` trait provides the additive identity and operations for a
/// `Number` type.
pub trait Addition: Copy + PartialOrd + Add<Output = Self> + AddAssign<Self> + Sum<Self> + Sub<Self, Output = Self> + SubAssign<Self> {
    /// The additive identity.
    const ZERO: Self;

    /// Returns the additive inverse of `self`.
    #[must_use]
    fn neg(self) -> Self {
        Self::ZERO - self
    }

    /// Returns the absolute value of `self`.
    #[must_use]
    fn abs(self) -> Self {
        if self < Self::ZERO { self.neg() } else { self }
    }

    /// Returns the absolute difference between `self` and `other`.
    #[must_use]
    fn abs_diff(self, other: Self) -> Self {
        if self < other { other - self } else { self - other }
    }
}

/// Macro to implement `Addition` for all integer types.
macro_rules! impl_addition {
    ($($ty:ty),*) => {
        $(
            impl Addition for $ty {
                const ZERO: Self = 0;
            }
        )*
    }
}

impl_addition!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

impl Addition for f32 {
    const ZERO: Self = 0.0;
}

impl Addition for f64 {
    const ZERO: Self = 0.0;
}

/// The `Multiplication` trait provides the multiplicative identity and
/// operations for a `Number` type.
pub trait Multiplication:
    Addition + Mul<Output = Self> + MulAssign<Self> + Div<Self, Output = Self> + DivAssign<Self> + Rem<Self, Output = Self> + RemAssign<Self>
{
    /// The multiplicative identity.
    const ONE: Self;

    /// Returns `self / 2`.
    #[must_use]
    fn half(self) -> Self {
        self / (Self::ONE + Self::ONE)
    }

    /// Returns `self * 2`.
    #[must_use]
    fn double(self) -> Self {
        self + self
    }

    /// Returns the multiplicative inverse of `self`.
    #[must_use]
    fn inv(self) -> Self {
        Self::ONE / self
    }

    /// Returns `self * a + b`, potentially as a fused multiply-add operation.
    #[must_use]
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Replace `self` with `self * a + b`, potentially as a fused
    /// multiply-add-assign operation.
    fn mul_add_assign(&mut self, a: Self, b: Self);

    /// Returns `self` raised to the power of `exp`.
    #[must_use]
    fn powi(self, exp: i32) -> Self;

    /// Returns the square of the number.
    #[must_use]
    fn square(self) -> Self {
        self * self
    }

    /// Returns the cube of the number.
    #[must_use]
    fn cube(self) -> Self {
        self.square() * self
    }
}

/// Macro to implement `Multiplication` for all floating-point types.
macro_rules! impl_multiplication_float {
    ($($ty:ty),*) => {
        $(
            impl Multiplication for $ty {
                const ONE: Self = 1.0;

                fn mul_add(self, a: Self, b: Self) -> Self {
                    self.mul_add(a, b)
                }

                fn mul_add_assign(&mut self, a: Self, b: Self) {
                    *self = self.mul_add(a, b);
                }

                fn powi(self, exp: i32) -> Self {
                    self.powi(exp)
                }
            }
        )*
    }
}

impl_multiplication_float!(f32, f64);

/// Macro to implement `IntNumber` for all integer types.
macro_rules! impl_multiplication_int {
    ($($ty:ty),*) => {
        $(
            impl Multiplication for $ty {
                const ONE: Self = 1;

                fn mul_add(self, a: Self, b: Self) -> Self {
                    self + a * b
                }

                fn mul_add_assign(&mut self, a: Self, b: Self) {
                    *self += a * b;
                }

                #[allow(clippy::cast_sign_loss)]
                fn powi(self, exp: i32) -> Self {
                    self.pow(exp as u32)
                }
            }
        )*
    }
}

impl_multiplication_int!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
