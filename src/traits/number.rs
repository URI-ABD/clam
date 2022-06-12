//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Numbers`.

use std::convert::TryInto;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::Sum;

use num_traits::Num;
use num_traits::NumCast;

/// Collections of `Numbers` can be used to calculate distances.
pub trait Number: Num + NumCast + Sum + Copy + Clone + PartialOrd + Send + Sync + Debug + Display {
    /// Returns the number of bytes used to store this number
    fn num_bytes() -> u8;

    /// Returns the number as a vec of bytes.
    ///
    /// This must be the inverse of from_bytes
    fn to_bytes(&self) -> Vec<u8>;

    /// Reconstructs the Number from its vec of bytes.
    ///
    /// This must be the inverse of to_bytes.
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Convers thhe number to an f64 for some helpful functions.
    fn as_f64(&self) -> f64;
}

macro_rules! impl_number {
    ($($ty:ty),*) => {
        $(
            impl Number for $ty {
                fn num_bytes() -> u8 {
                    (0 as $ty).to_be_bytes().to_vec().len() as u8
                }

                fn to_bytes(&self) -> Vec<u8> {
                    <$ty>::to_be_bytes(*self).to_vec()
                }

                fn from_bytes(bytes: &[u8]) -> $ty {
                    let (value, _) = bytes.split_at(std::mem::size_of::<$ty>());
                    <$ty>::from_be_bytes(value.try_into().unwrap())
                }

                fn as_f64(&self) -> f64 {
                    *self as f64
                }
            }
        )*
    }
}

impl_number!(f32, f64, u8, i8, u16, i16, u32, i32, u64, i64, isize, usize);
