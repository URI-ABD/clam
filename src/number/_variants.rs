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
pub trait Float: Number {
    /// Returns the square root of a `Float`.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Returns the cube root of a `Float`.
    #[must_use]
    fn cbrt(self) -> Self;

    /// Returns the machine epsilon for a `Float`.
    fn epsilon() -> Self;

    /// Returns the inverse square root of a `Float`.
    ///
    /// This is equivalent to `1.0 / self.sqrt()`. It uses the fast quake inverse
    /// square root algorithm. The implementations are adapted from those available
    /// in [this crate](https://github.com/emkw/rust-fast_inv_sqrt).
    ///
    /// Benchmarks for this implementation can be found in benches/inv-sqrt.rs.
    /// They show this implementation to be about ~4.5x faster than using
    /// `1.0 / self.sqrt()` or `self.sqrt().recip()` and accurate to `1e-6` for
    /// both `f32` and `f64`.
    ///
    /// # References
    ///
    /// - [Fast Inverse Square Root](https://en.wikipedia.org/wiki/Fast_inverse_square_root)
    /// - [Quake III Arena Fast InvSqrt()](https://www.youtube.com/watch?v=p8u_k2LIZyo)
    #[must_use]
    fn inv_sqrt(self) -> Self;

    /// Returns `self` raised to the power of `exp`.
    #[must_use]
    fn powf(self, exp: Self) -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> Self {
        // libm::sqrtf(self)  // no-std
        self.sqrt()
    }

    fn cbrt(self) -> Self {
        // libm::cbrtf(self)  // no-std
        self.cbrt()
    }

    fn epsilon() -> Self {
        Self::EPSILON
    }

    fn inv_sqrt(self) -> Self {
        if self == 0.0 {
            return Self::NAN;
        } else if self == Self::INFINITY {
            return 0.0;
        } else if self < Self::MIN_POSITIVE {
            return Self::INFINITY;
        }

        let x2 = self * 0.5;
        let i = 0x5f_375_a86 - (self.to_bits() >> 1);

        let mut y = Self::from_bits(i);

        // More iterations can be added for higher precision.
        // This achieves a precision of 1e-6 with no loss of speed.
        y *= (x2 * y).mul_add(-y, 1.5);
        y *= (x2 * y).mul_add(-y, 1.5);

        y * (x2 * y).mul_add(-y, 1.5)
    }

    fn powf(self, exp: Self) -> Self {
        // libm::powf(self, exp)  // no-std
        self.powf(exp)
    }
}

impl Float for f64 {
    fn sqrt(self) -> Self {
        // libm::sqrt(self)  // no-std
        self.sqrt()
    }

    fn cbrt(self) -> Self {
        // libm::cbrt(self)  // no-std
        self.cbrt()
    }

    fn epsilon() -> Self {
        Self::EPSILON
    }

    fn inv_sqrt(self) -> Self {
        if self == 0.0 {
            return Self::NAN;
        } else if self == Self::INFINITY {
            return 0.0;
        } else if self < Self::MIN_POSITIVE {
            return Self::INFINITY;
        }

        let x2 = self * 0.5;
        let i = 0x5_fe6_eb5_0c7_b53_7a9 - (self.to_bits() >> 1);

        let mut y = Self::from_bits(i);

        // More iterations can be added for higher precision.
        // This achieves a precision of 1e-6 with no loss of speed.
        y *= (x2 * y).mul_add(-y, 1.5);
        y *= (x2 * y).mul_add(-y, 1.5);

        y * (x2 * y).mul_add(-y, 1.5)
    }

    fn powf(self, exp: Self) -> Self {
        // libm::pow(self, exp)  // no-std
        self.powf(exp)
    }
}
