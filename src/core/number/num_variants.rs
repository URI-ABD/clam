use super::Number;

pub trait IntNumber: Number {}

macro_rules! impl_int_number {
    ($($ty:ty),*) => {
        $(
            impl IntNumber for $ty {}
        )*
    }
}

impl_int_number!(u8, i8, u16, i16, u32, i32, u64, i64);

pub trait IIntNumber: Number {}

macro_rules! impl_iint_number {
    ($($ty:ty),*) => {
        $(
            impl IIntNumber for $ty {}
        )*
    }
}

impl_iint_number!(i8, i16, i32, i64);

pub trait UIntNumber: Number {}

macro_rules! impl_uint_number {
    ($($ty:ty),*) => {
        $(
            impl UIntNumber for $ty {}
        )*
    }
}

impl_uint_number!(u8, u16, u32, u64);

pub trait FloatNumber: Number {}

macro_rules! impl_float_number {
    ($($ty:ty),*) => {
        $(
            impl FloatNumber for $ty {}
        )*
    }
}

impl_float_number!(f32, f64);
