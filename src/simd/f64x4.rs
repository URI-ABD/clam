use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_ty!(F64x4, f64, f64, f64, f64);
impl_minimal!(F64x4, f64, 4, x0, x1, x2, x3);

impl F64x4 {
    /// Create a new `F64x4` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 4 elements long.
    pub fn from_slice(slice: &[f64]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(slice[0], slice[1], slice[2], slice[3])
    }

    pub fn horizontal_add(self) -> f64 {
        self.0 + self.1 + self.2 + self.3
    }
}

impl_op4!(Mul, mul, F64x4, *);
impl_op4!(assn MulAssign, mul_assign, F64x4, *=);
impl_op4!(Div, div, F64x4, /);
impl_op4!(assn DivAssign, div_assign, F64x4, /=);
impl_op4!(Add, add, F64x4, +);
impl_op4!(assn AddAssign, add_assign, F64x4, +=);
impl_op4!(Sub, sub, F64x4, -);
impl_op4!(assn SubAssign, sub_assign, F64x4, -=);

impl_distances!(F64x4, f64);
