//! A `Number` is a general numeric type.
//!
//! We calculate distances over collections of `Number`s.
//! Distance values are also represented as `Number`s.

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
}

macro_rules! impl_number {
    ($($ty:ty),*) => {
        $(
            impl Number for $ty {
                fn num_bytes() -> usize {
                    (0 as $ty).to_be_bytes().to_vec().len()
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

                fn as_f32(&self) -> f32 {
                    *self as f32
                }

                fn as_i64(&self) -> i64 {
                    *self as i64
                }

                fn as_u64(&self) -> u64 {
                    *self as u64
                }
            }
        )*
    }
}

impl_number!(f32, f64, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, isize, usize);
