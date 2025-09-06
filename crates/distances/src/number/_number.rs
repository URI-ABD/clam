//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Number`s.
//! Distance values are also represented as `Number`s.

use rand::RngExt;

use core::{
    fmt::{Debug, Display},
    str::FromStr,
};

use super::{Addition, Multiplication};

/// Collections of `Number`s can be used to calculate distances.
pub trait Number: Addition + Multiplication + PartialEq + Clone + Send + Sync + Debug + Display + Default + FromStr {
    /// The minimum possible value.
    const MIN: Self;

    /// The maximum possible value.
    const MAX: Self;

    /// The difference between `ONE` and the next largest representable number.
    const EPSILON: Self;

    /// The number of bytes used to represent this type.
    const NUM_BYTES: usize;

    /// The additive identity.
    #[deprecated(since = "1.8.0", note = "Use `Number::ZERO` instead")]
    #[must_use]
    fn zero() -> Self {
        Self::ZERO
    }

    /// The multiplicative identity.
    #[deprecated(since = "1.8.0", note = "Use `Number::ONE` instead")]
    #[must_use]
    fn one() -> Self {
        Self::ONE
    }

    /// The minimum possible value.
    #[deprecated(since = "1.8.0", note = "Use `Number::MIN` instead")]
    #[must_use]
    fn min_value() -> Self {
        Self::MIN
    }

    /// The maximum possible value.
    #[deprecated(since = "1.8.0", note = "Use `Number::MAX` instead")]
    #[must_use]
    fn max_value() -> Self {
        Self::MAX
    }

    /// The difference between `ONE` and the next largest representable number.
    #[deprecated(since = "1.8.0", note = "Use `Number::EPSILON` instead")]
    #[must_use]
    fn epsilon() -> Self {
        Self::EPSILON
    }

    /// Casts a number to `Self`. This may be a lossy conversion.
    fn from<T: Number>(n: T) -> Self;

    /// Returns the number as a `f32`. This may be a lossy conversion.
    fn as_f32(self) -> f32;

    /// Returns the number as a `f64`. This may be a lossy conversion.
    fn as_f64(self) -> f64;

    /// Returns the number as a `usize`. This may be a lossy conversion.
    #[allow(clippy::cast_possible_truncation)]
    fn as_usize(self) -> usize {
        self.as_u64() as usize
    }

    /// Returns the number as a `isize`. This may be a lossy conversion.
    #[allow(clippy::cast_possible_truncation)]
    fn as_isize(self) -> isize {
        self.as_i64() as isize
    }

    /// Returns the number as a `u64`. This may be a lossy conversion.
    fn as_u64(self) -> u64;

    /// Returns the number as an `i64`. This may be a lossy conversion.
    fn as_i64(self) -> i64;

    /// Returns the number as a `u32`. This may be a lossy conversion.
    fn as_u32(self) -> u32;

    /// Returns the number as an `i32`. This may be a lossy conversion.
    fn as_i32(self) -> i32;

    /// Returns the number of bytes used to represent a `Number`.
    #[deprecated(since = "1.8.0", note = "Use `Number::NUM_BYTES` instead")]
    #[must_use]
    fn num_bytes() -> usize {
        Self::NUM_BYTES
    }

    /// Reads a `Number` from little endian bytes.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Converts a `Number` to little endian bytes.
    fn to_le_bytes(self) -> Vec<u8>;

    /// Reads a `Number` from big endian bytes.
    fn from_be_bytes(bytes: &[u8]) -> Self;

    /// Converts a `Number` to big endian bytes.
    fn to_be_bytes(self) -> Vec<u8>;

    /// Returns the name of the type.
    #[must_use]
    fn type_name<'a>() -> &'a str {
        core::any::type_name::<Self>()
    }

    /// Returns a random `Number`.
    fn next_random<R: rand::Rng>(rng: &mut R) -> Self;

    /// Returns a total ordering of the number.
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering;

    /// Returns the smaller of two numbers.
    #[must_use]
    fn min(self, other: Self) -> Self {
        if self < other { self } else { other }
    }

    /// Returns the larger of two numbers.
    #[must_use]
    fn max(self, other: Self) -> Self {
        if self > other { self } else { other }
    }
}

impl Number for f32 {
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    const EPSILON: Self = Self::EPSILON;
    const NUM_BYTES: usize = core::mem::size_of::<Self>();

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

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn as_u32(self) -> u32 {
        self as u32
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_i32(self) -> i32 {
        self as i32
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; 4];
        ty_bytes.copy_from_slice(bytes);
        Self::from_le_bytes(ty_bytes)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; 4];
        ty_bytes.copy_from_slice(bytes);
        Self::from_be_bytes(ty_bytes)
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
        rng.random()
    }

    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.total_cmp(other)
    }
}

impl Number for f64 {
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    const EPSILON: Self = Self::EPSILON;
    const NUM_BYTES: usize = core::mem::size_of::<Self>();

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

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn as_u32(self) -> u32 {
        self as u32
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_i32(self) -> i32 {
        self as i32
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; 8];
        ty_bytes.copy_from_slice(bytes);
        Self::from_le_bytes(ty_bytes)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; 8];
        ty_bytes.copy_from_slice(bytes);
        Self::from_be_bytes(ty_bytes)
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
        rng.random()
    }

    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.total_cmp(other)
    }
}

/// A macro to implement the `Number` trait for primitive types.
macro_rules! impl_number_iint {
    ($($ty:ty),*) => {
        $(
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, clippy::cast_lossless)]
            impl Number for $ty {
                const MAX: Self = <$ty>::MAX;
                const MIN: Self = <$ty>::MIN;
                const EPSILON: Self = 1;
                const NUM_BYTES: usize = core::mem::size_of::<$ty>();

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

                fn as_u32(self) -> u32 {
                    self as u32
                }

                fn as_i32(self) -> i32 {
                    self as i32
                }

                fn from_le_bytes(bytes: &[u8]) -> Self {
                    let mut ty_bytes = [0_u8; core::mem::size_of::<$ty>()];
                    ty_bytes.copy_from_slice(bytes);
                    Self::from_le_bytes(ty_bytes)
                }

                fn to_le_bytes(self) -> Vec<u8> {
                    self.to_le_bytes().to_vec()
                }

                fn from_be_bytes(bytes: &[u8]) -> Self {
                    let mut ty_bytes = [0_u8; core::mem::size_of::<$ty>()];
                    ty_bytes.copy_from_slice(bytes);
                    Self::from_be_bytes(ty_bytes)
                }

                fn to_be_bytes(self) -> Vec<u8> {
                    self.to_be_bytes().to_vec()
                }

                fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
                    rng.random()
                }

                fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
                    self.cmp(other)
                }
            }
        )*
    }
}

impl_number_iint!(i8, i16, i32, i64, i128);

/// A macro to implement the `Number` trait for primitive types.
macro_rules! impl_number_uint {
    ($($ty:ty),*) => {
        $(
            #[allow(clippy::cast_possible_truncation, clippy::cast_lossless, clippy::cast_precision_loss)]
            impl Number for $ty {
                const MAX: Self = <$ty>::MAX;
                const MIN: Self = <$ty>::MIN;
                const EPSILON: Self = 1;
                const NUM_BYTES: usize = core::mem::size_of::<$ty>();

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

                #[allow(clippy::cast_possible_wrap)]
                fn as_i64(self) -> i64 {
                    self as i64
                }

                fn as_u32(self) -> u32 {
                    self as u32
                }

                #[allow(clippy::cast_possible_wrap)]
                fn as_i32(self) -> i32 {
                    self as i32
                }

                fn from_le_bytes(bytes: &[u8]) -> Self {
                    let mut ty_bytes = [0_u8; core::mem::size_of::<$ty>()];
                    ty_bytes.copy_from_slice(bytes);
                    Self::from_le_bytes(ty_bytes)
                }

                fn to_le_bytes(self) -> Vec<u8> {
                    self.to_le_bytes().to_vec()
                }

                fn from_be_bytes(bytes: &[u8]) -> Self {
                    let mut ty_bytes = [0_u8; core::mem::size_of::<$ty>()];
                    ty_bytes.copy_from_slice(bytes);
                    Self::from_be_bytes(ty_bytes)
                }

                fn to_be_bytes(self) -> Vec<u8> {
                    self.to_be_bytes().to_vec()
                }

                fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
                    rng.random()
                }

                fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
                    self.cmp(other)
                }
            }
        )*
    }
}

impl_number_uint!(u8, u16, u32, u64, u128);

impl Number for usize {
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    const EPSILON: Self = 1;
    const NUM_BYTES: Self = core::mem::size_of::<Self>();

    fn from<T: Number>(n: T) -> Self {
        n.as_usize()
    }

    #[allow(clippy::cast_precision_loss)]
    fn as_f32(self) -> f32 {
        self as f32
    }

    #[allow(clippy::cast_precision_loss)]
    fn as_f64(self) -> f64 {
        self as f64
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_u64(self) -> u64 {
        self as u64
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn as_i64(self) -> i64 {
        self as i64
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_u32(self) -> u32 {
        self as u32
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn as_i32(self) -> i32 {
        self as i32
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; core::mem::size_of::<Self>()];
        ty_bytes.copy_from_slice(bytes);
        Self::from_le_bytes(ty_bytes)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; core::mem::size_of::<Self>()];
        ty_bytes.copy_from_slice(bytes);
        Self::from_be_bytes(ty_bytes)
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
        rng.random::<u64>() as Self
    }

    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cmp(other)
    }
}

impl Number for isize {
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    const EPSILON: Self = 1;
    const NUM_BYTES: usize = core::mem::size_of::<Self>();

    fn from<T: Number>(n: T) -> Self {
        n.as_isize()
    }

    #[allow(clippy::cast_precision_loss)]
    fn as_f32(self) -> f32 {
        self as f32
    }

    #[allow(clippy::cast_precision_loss)]
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

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn as_u32(self) -> u32 {
        self as u32
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_i32(self) -> i32 {
        self as i32
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; core::mem::size_of::<Self>()];
        ty_bytes.copy_from_slice(bytes);
        Self::from_le_bytes(ty_bytes)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut ty_bytes = [0_u8; core::mem::size_of::<Self>()];
        ty_bytes.copy_from_slice(bytes);
        Self::from_be_bytes(ty_bytes)
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
        rng.random::<i64>() as Self
    }

    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cmp(other)
    }
}
