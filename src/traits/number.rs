//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Number`s.
//! Distance values are also represented as `Number`s.

use std::convert::TryInto;

/// Collections of `Number`s can be used to calculate distances.
pub trait Number:
    num_traits::Num
    + num_traits::NumCast
    + std::iter::Sum
    + Copy
    + Clone
    + PartialOrd
    + Send
    + Sync
    + std::fmt::Debug
    + std::fmt::Display
    + serde::Serialize
    + serde::Deserialize<'static>
{
    /// Returns the number of bytes used to store this number
    fn num_bytes() -> u8;

    /// Returns the number as a vec of bytes.
    ///
    /// This must be the inverse of from_le_bytes
    fn to_le_bytes(&self) -> Vec<u8>;
    fn to_be_bytes(&self) -> Vec<u8>;

    /// Reconstructs the Number from its vec of bytes.
    ///
    /// This must be the inverse of to_le_bytes.
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String>;
    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String>;

    fn as_f64(&self) -> f64;
    // fn as_f32(&self) -> f32;
}

macro_rules! impl_number {
    ($($ty:ty),*) => {
        $(
            impl Number for $ty {
                fn num_bytes() -> u8 {
                    (0 as $ty).to_be_bytes().to_vec().len() as u8
                }

                fn to_le_bytes(&self) -> Vec<u8> {
                    <$ty>::to_le_bytes(*self).to_vec()
                }

                fn to_be_bytes(&self) -> Vec<u8> {
                    <$ty>::to_be_bytes(*self).to_vec()
                }

                fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
                    let (value, _) = bytes.split_at(std::mem::size_of::<$ty>());
                    Ok(<$ty>::from_le_bytes(value.try_into().map_err(|reason| {
                        format!("Could not construct Number from bytes {:?} because {}", value, reason)
                    })?))
                }

                fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
                    let (value, _) = bytes.split_at(std::mem::size_of::<$ty>());
                    Ok(<$ty>::from_be_bytes(value.try_into().map_err(|reason| {
                        format!("Could not construct Number from bytes {:?} because {}", value, reason)
                    })?))
                }

                fn as_f64(&self) -> f64 {
                    *self as f64
                }

                // fn as_f32(&self) -> f32 {
                //     *self as f32
                // }
            }
        )*
    }
}

impl_number!(f32, f64, u8, i8, u16, i16, u32, i32, u64, i64, isize, usize);
