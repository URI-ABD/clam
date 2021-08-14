//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Numbers`.

use std::convert::TryInto;
use std::fmt::{Debug, Display};
use std::iter::Sum;

use ndarray_npy::{ReadableElement, WritableElement};
use num_traits::{Num, NumCast};

/// Collections of `Numbers` can be used to calculate distances.
pub trait Number: Num + NumCast + Sum + Copy + Clone + PartialOrd + Send + Sync + Debug + Display + ReadableElement + WritableElement {
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
            }
        )*
    }
}

impl_number!(f32, f64, u8, i8, u16, i16, u32, i32, u64, i64);
