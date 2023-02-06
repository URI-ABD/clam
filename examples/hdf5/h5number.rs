pub trait H5Number: clam::Number + hdf5::H5Type {}

macro_rules! impl_h5number {
    ($($ty:ty),*) => {
        $(
            impl H5Number for $ty {}
        )*
    }
}

impl_h5number!(f32, f64, u8, i8, u16, i16, u32, i32, u64, i64, isize, usize);
