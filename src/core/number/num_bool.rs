use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray};
use ndarray_npy::{ReadableElement, WritableElement};

use super::Number;

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

    pub fn as_bool(&self) -> bool {
        self.0.is_one()
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

impl<T: Number> num_traits::ToPrimitive for NumBool<T> {
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

impl<T: Number> num_traits::Num for NumBool<T> {
    type FromStrRadixErr = <T as num_traits::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str, radix).map(|v| Self(v))
    }
}

impl<T: Number> ReadableElement for NumBool<T> {
    fn read_to_end_exact_vec<R: std::io::Read>(
        reader: R,
        type_desc: &py_literal::Value,
        len: usize,
    ) -> Result<Vec<Self>, ndarray_npy::ReadDataError> {
        bool::read_to_end_exact_vec(reader, type_desc, len)
            .map(|v| v.into_iter().map(Self::from_bool).collect::<Vec<_>>())
    }
}

unsafe impl<T: Number> WritableElement for NumBool<T> {
    fn type_descriptor() -> py_literal::Value {
        bool::type_descriptor()
    }

    fn write<W: std::io::Write>(&self, writer: W) -> Result<(), ndarray_npy::WriteDataError> {
        self.as_bool().write(writer)
    }

    fn write_slice<W: std::io::Write>(slice: &[Self], writer: W) -> Result<(), ndarray_npy::WriteDataError> {
        let v = slice.iter().map(|b| b.as_bool()).collect::<Vec<_>>();
        bool::write_slice(&v, writer)
    }
}

impl<T: Number> Number for NumBool<T> {
    fn num_bytes() -> usize {
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

    fn as_arrow_array(slice: &[Self]) -> ArrayRef {
        Arc::new(BooleanArray::from_iter(slice.iter().map(|b| Some(b.as_bool()))))
    }
}
