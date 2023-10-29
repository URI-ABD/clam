use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_ty!(F64x8, f64, f64, f64, f64, f64, f64, f64, f64);
impl_minimal!(F64x8, f64, 8, x0, x1, x2, x3, x4, x5, x6, x7);

impl F64x8 {
    /// Create a new `F64x8` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 8 elements long.
    pub fn from_slice(slice: &[f64]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        )
    }

    pub fn horizontal_add(self) -> f64 {
        self.0 + self.1 + self.2 + self.3 + self.4 + self.5 + self.6 + self.7
    }
}

impl_op8!(Mul, mul, F64x8, *);
impl_op8!(assn MulAssign, mul_assign, F64x8, *=);
impl_op8!(Div, div, F64x8, /);
impl_op8!(assn DivAssign, div_assign, F64x8, /=);
impl_op8!(Add, add, F64x8, +);
impl_op8!(assn AddAssign, add_assign, F64x8, +=);
impl_op8!(Sub, sub, F64x8, -);
impl_op8!(assn SubAssign, sub_assign, F64x8, -=);

impl_distances!(F64x8, f64);
