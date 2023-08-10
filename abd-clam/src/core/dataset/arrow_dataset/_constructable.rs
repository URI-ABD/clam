use distances::Number;

pub trait ConstructableNumber: Number {
    fn from_ne_bytes(bytes: &[u8]) -> Self;
}

/// A macro to implement the `Number` trait for primitive types.
macro_rules! impl_constructable {
    ($($ty:ty),*) => {
        $(
            impl ConstructableNumber for $ty {
                fn from_ne_bytes(bytes: &[u8]) -> Self {
                    <$ty>::from_ne_bytes(bytes[0..std::mem::size_of::<$ty>()].try_into().unwrap())
                }
            }
        )*
    }
}
impl_constructable!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);
