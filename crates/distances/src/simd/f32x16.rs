use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_ty!(F32x16, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32);
impl_minimal!(
    F32x16, f32, 16, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
);

impl F32x16 {
    /// Create a new `F32x16` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 16 elements long.
    pub fn from_slice(slice: &[f32]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            slice[8], slice[9], slice[10], slice[11], slice[12], slice[13], slice[14], slice[15],
        )
    }

    pub fn horizontal_add(self) -> f32 {
        self.0
            + self.1
            + self.2
            + self.3
            + self.4
            + self.5
            + self.6
            + self.7
            + self.8
            + self.9
            + self.10
            + self.11
            + self.12
            + self.13
            + self.14
            + self.15
    }
}

impl_op16!(Mul, mul, F32x16, *);
impl_op16!(assn MulAssign, mul_assign, F32x16, *=);
impl_op16!(Div, div, F32x16, /);
impl_op16!(assn DivAssign, div_assign, F32x16, /=);
impl_op16!(Add, add, F32x16, +);
impl_op16!(assn AddAssign, add_assign, F32x16, +=);
impl_op16!(Sub, sub, F32x16, -);
impl_op16!(assn SubAssign, sub_assign, F32x16, -=);
impl_distances!(F32x16, f32);
