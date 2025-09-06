//! `NumBool` is a `Number` that can be used as a boolean.

use rand::RngExt;

use crate::Number;

use super::{Addition, Multiplication};

/// A `Number` that can be used as a boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Bool(u8);

impl Bool {
    /// Creates a new `NumBool` from a `bool`.
    #[must_use]
    pub const fn from_bool(b: bool) -> Self {
        Self(if b { 1 } else { 0 })
    }

    /// Returns the `NumBool` as a `bool`.
    #[must_use]
    pub const fn as_bool(&self) -> bool {
        self.0 == 1
    }
}

impl Addition for Bool {
    const ZERO: Self = Self(0);
}

impl Multiplication for Bool {
    const ONE: Self = Self(1);

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }

    fn mul_add_assign(&mut self, a: Self, b: Self) {
        self.0.mul_add_assign(a.0, b.0);
    }

    fn powi(self, _: i32) -> Self {
        if self.0 == 0 { Self(0) } else { Self(1) }
    }
}

impl core::fmt::Display for Bool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_bool())
    }
}

impl core::iter::Sum for Bool {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if iter.any(|v| v.as_bool()) { Self(1) } else { Self(0) }
    }
}

impl core::ops::Add for Bool {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl core::ops::AddAssign for Bool {
    fn add_assign(&mut self, rhs: Self) {
        self.0.add_assign(rhs.0);
    }
}

impl core::ops::Mul for Bool {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl core::ops::MulAssign for Bool {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(rhs.0);
    }
}

impl core::ops::Rem for Bool {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0.rem(rhs.0))
    }
}

impl core::ops::RemAssign for Bool {
    fn rem_assign(&mut self, rhs: Self) {
        self.0.rem_assign(rhs.0);
    }
}

impl core::ops::Div for Bool {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0))
    }
}

impl core::ops::DivAssign for Bool {
    fn div_assign(&mut self, rhs: Self) {
        self.0.div_assign(rhs.0);
    }
}

impl core::ops::Sub for Bool {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl core::ops::SubAssign for Bool {
    fn sub_assign(&mut self, rhs: Self) {
        self.0.sub_assign(rhs.0);
    }
}

impl core::str::FromStr for Bool {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let b = s.parse::<bool>().map_err(|e| e.to_string())?;
        Ok(Self::from_bool(b))
    }
}

impl Number for Bool {
    const MAX: Self = Self(1);
    const MIN: Self = Self(0);
    const EPSILON: Self = Self(1);
    const NUM_BYTES: usize = 1;

    fn from<T: Number>(n: T) -> Self {
        if n == T::ZERO { Self::ZERO } else { Self::ONE }
    }

    fn as_f32(self) -> f32 {
        self.0.as_f32()
    }

    fn as_f64(self) -> f64 {
        self.0.as_f64()
    }

    fn as_u64(self) -> u64 {
        self.0.as_u64()
    }

    fn as_i64(self) -> i64 {
        self.0.as_i64()
    }

    fn as_u32(self) -> u32 {
        self.0.as_u32()
    }

    fn as_i32(self) -> i32 {
        self.0.as_i32()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self::from_bool(bytes[0] != 0)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes)
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_le_bytes()
    }

    fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
        Self(rng.random())
    }

    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
