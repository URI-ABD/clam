//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Numbers`.

use std::convert::TryInto;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::Sum;

use easy_cast::Conv;
use ndarray_npy::ReadableElement;
use ndarray_npy::WritableElement;
use num_traits::Num;
use num_traits::NumCast;

/// Collections of `Numbers` can be used to calculate distances.
pub trait Number:
    Num
    + NumCast
    + Sum
    + Copy
    + Clone
    + PartialOrd
    + Send
    + Sync
    + Debug
    + Display
    + ReadableElement
    + WritableElement
    + Conv<Self>
{
    /// Returns the number of bytes used to store this number
    fn num_bytes() -> u8;

    /// Returns the number as a vec of bytes.
    ///
    /// This must be the inverse of from_bytes
    fn to_bytes(self) -> Vec<u8>;

    /// Reconstructs the Number from its vec of bytes.
    ///
    /// This must be the inverse of to_bytes.
    fn from_bytes(bytes: &[u8]) -> Self;

    fn to_f64(self) -> f64;
}

impl Number for f32 {
    fn num_bytes() -> u8 {
        4
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<f32>());
        f32::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for f64 {
    fn num_bytes() -> u8 {
        8
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<f64>());
        f64::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        self
    }
}

impl Number for u8 {
    fn num_bytes() -> u8 {
        1
    }

    fn to_bytes(self) -> Vec<u8> {
        vec![self]
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0]
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for u16 {
    fn num_bytes() -> u8 {
        2
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<u16>());
        u16::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for u32 {
    fn num_bytes() -> u8 {
        4
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<u32>());
        u32::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for u64 {
    fn num_bytes() -> u8 {
        8
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<u64>());
        u64::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for i8 {
    fn num_bytes() -> u8 {
        1
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<i8>());
        i8::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for i16 {
    fn num_bytes() -> u8 {
        2
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<i16>());
        i16::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for i32 {
    fn num_bytes() -> u8 {
        4
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<i32>());
        i32::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}

impl Number for i64 {
    fn num_bytes() -> u8 {
        8
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let (value, _) = bytes.split_at(std::mem::size_of::<i64>());
        i64::from_be_bytes(value.try_into().unwrap())
    }

    fn to_f64(self) -> f64 {
        f64::conv(self)
    }
}
