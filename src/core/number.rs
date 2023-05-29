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
    fn num_bytes() -> u8;

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

pub trait IntNumber: Number {}

macro_rules! impl_int_number {
    ($($ty:ty),*) => {
        $(
            impl IntNumber for $ty {}
        )*
    }
}

impl_int_number!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, isize, usize);

pub trait IIntNumber: Number {}

macro_rules! impl_iint_number {
    ($($ty:ty),*) => {
        $(
            impl IIntNumber for $ty {}
        )*
    }
}

impl_iint_number!(i8, i16, i32, i64, i128, isize);

pub trait UIntNumber: Number {}

macro_rules! impl_uint_number {
    ($($ty:ty),*) => {
        $(
            impl UIntNumber for $ty {}
        )*
    }
}

impl_uint_number!(u8, u16, u32, u64, u128, usize);

pub trait FloatNumber: Number {}

macro_rules! impl_float_number {
    ($($ty:ty),*) => {
        $(
            impl FloatNumber for $ty {}
        )*
    }
}

impl_float_number!(f32, f64);

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    // serde::Serialize,
    // serde::Deserialize<'static>,
)]
pub struct NumBool<T: Number>(T);

impl<T: Number> NumBool<T> {
    pub fn from_bool(b: bool) -> Self {
        Self(if b { T::one() } else { T::zero() })
    }
}

impl<T: Number> std::fmt::Display for NumBool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0 != T::zero())
    }
}

impl<T: Number> std::iter::Sum for NumBool<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(NumBool(T::zero()), |mut acc, v| {
            acc.0 += v.0;
            acc
        })
    }
}

impl<U: Number> num_traits::ToPrimitive for NumBool<U> {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0.as_i64())
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.0.as_u64())
    }
}

impl<U: Number> num_traits::NumCast for NumBool<U> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        U::from(n).map(|v| Self(v))
    }
}

impl<T: Number> std::ops::Add for NumBool<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl<T: Number> std::ops::AddAssign for NumBool<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0.add_assign(rhs.0)
    }
}

impl<T: Number> num_traits::Zero for NumBool<T> {
    fn zero() -> Self {
        Self(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<T: Number> std::ops::Mul for NumBool<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl<T: Number> std::ops::MulAssign for NumBool<T> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(rhs.0)
    }
}

impl<T: Number> num_traits::One for NumBool<T> {
    fn one() -> Self {
        Self(T::one())
    }
}

impl<T: Number> std::ops::Rem for NumBool<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0.rem(rhs.0))
    }
}

impl<T: Number> std::ops::RemAssign for NumBool<T> {
    fn rem_assign(&mut self, rhs: Self) {
        self.0.rem_assign(rhs.0)
    }
}

impl<T: Number> std::ops::Div for NumBool<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0))
    }
}

impl<T: Number> std::ops::DivAssign for NumBool<T> {
    fn div_assign(&mut self, rhs: Self) {
        self.0.div_assign(rhs.0)
    }
}

impl<T: Number> std::ops::Sub for NumBool<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl<T: Number> std::ops::SubAssign for NumBool<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0.sub_assign(rhs.0)
    }
}

impl<U: Number> num_traits::Num for NumBool<U> {
    type FromStrRadixErr = <U as num_traits::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        U::from_str_radix(str, radix).map(|v| Self(v))
    }
}

impl<T: Number> Number for NumBool<T> {
    fn num_bytes() -> u8 {
        T::num_bytes()
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes()
    }

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        T::from_le_bytes(bytes).map(|v| Self(v))
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        self.0.to_be_bytes()
    }

    fn from_be_bytes(bytes: &[u8]) -> Result<Self, String> {
        T::from_be_bytes(bytes).map(|v| Self(v))
    }

    fn as_f64(&self) -> f64 {
        self.0.as_f64()
    }

    fn as_f32(&self) -> f32 {
        self.0.as_f32()
    }

    fn as_i64(&self) -> i64 {
        self.0.as_i64()
    }

    fn as_u64(&self) -> u64 {
        self.0.as_u64()
    }
}
