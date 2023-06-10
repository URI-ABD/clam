//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Number`s.
//! Distance values are also represented as `Number`s.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array,
    UInt64Array, UInt8Array,
};
use ndarray_npy::{ReadableElement, WritableElement};

// TODO: See if we can instead rely on a trait from `num_traits`.
/// Collections of `Number`s can be used to calculate distances.
pub trait Number:
    num_traits::Num
    + num_traits::ToPrimitive
    + num_traits::NumCast
    + num_traits::Zero
    + std::ops::Add
    + std::ops::AddAssign
    + std::iter::Sum
    + std::ops::Sub
    + std::ops::SubAssign
    + num_traits::One
    + std::ops::Mul
    + std::ops::MulAssign
    + std::ops::Div
    + std::ops::DivAssign
    + std::ops::Rem
    + std::ops::RemAssign
    + Copy
    + Clone
    + PartialOrd
    + Send
    + Sync
    + std::fmt::Debug
    + std::fmt::Display
    + ReadableElement
    + WritableElement
{
    /// Returns the number of bytes used to store this number
    fn num_bytes() -> usize;

    /// Returns the number as a vec of little-endian bytes.
    ///
    /// This must be the inverse of `from_le_bytes`.
    fn to_le_bytes(&self) -> Vec<u8>;

    /// Reconstructs the Number from its vec of bytes.
    ///
    /// This must be the inverse of to_le_bytes.
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String>;

    /// Returns the number as a vec of big-endian bytes.
    ///
    /// This must be the inverse of `from_be_bytes`.
    fn to_be_bytes(&self) -> Vec<u8>;
    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String>;

    // TODO: See if any/all of these can ba removed
    fn as_f64(&self) -> f64;
    fn as_f32(&self) -> f32;
    fn as_i64(&self) -> i64;
    fn as_u64(&self) -> u64;

    fn as_arrow_array(slice: &[Self]) -> ArrayRef;
}

impl Number for u8 {
    fn num_bytes() -> usize {
        1
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        vec![*self]
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        vec![*self]
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() == 1 {
            Ok(bytes[0])
        } else {
            Err(format!(
                "Incorrect number of bytes. Expected 1 byte but got {}",
                bytes.len()
            ))
        }
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() == 1 {
            Ok(bytes[0])
        } else {
            Err(format!(
                "Incorrect number of bytes. Expected 1 byte but got {}",
                bytes.len()
            ))
        }
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(UInt8Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for i8 {
    fn num_bytes() -> usize {
        1
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(Int8Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for u16 {
    fn num_bytes() -> usize {
        2
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(UInt16Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for i16 {
    fn num_bytes() -> usize {
        2
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(Int16Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for u32 {
    fn num_bytes() -> usize {
        4
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(UInt32Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for i32 {
    fn num_bytes() -> usize {
        4
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(Int32Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for u64 {
    fn num_bytes() -> usize {
        8
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(UInt64Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for i64 {
    fn num_bytes() -> usize {
        8
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(Int64Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for f32 {
    fn num_bytes() -> usize {
        4
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn as_f32(&self) -> f32 {
        *self
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(Float32Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}

impl Number for f64 {
    fn num_bytes() -> usize {
        8
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        Self::to_le_bytes(*self).to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        Self::to_be_bytes(*self).to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_le_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (value, _) = bytes.split_at(Self::num_bytes());
        Ok(<Self>::from_be_bytes(value.try_into().map_err(|reason| {
            format!("Could not construct Number from bytes {:?} because {}", value, reason)
        })?))
    }

    fn as_f64(&self) -> f64 {
        *self
    }

    fn as_f32(&self) -> f32 {
        *self as f32
    }

    fn as_i64(&self) -> i64 {
        *self as i64
    }

    fn as_u64(&self) -> u64 {
        *self as u64
    }

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(Float64Array::from_iter(slice.iter().map(|&v| Some(v))))
    }
}
