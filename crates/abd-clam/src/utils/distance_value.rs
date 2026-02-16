//! A trait for types that can be used as distance values.

use core::fmt::{Debug, Display};

/// A trait for types that can be used as distance values in clustering algorithms.
///
/// We provide a blanket implementation for all types that satisfy the trait bounds. This includes all primitive numeric types.
#[must_use]
pub trait DistanceValue:
    PartialEq
    + PartialOrd
    + Copy
    + Display
    + Debug
    + Default
    + num_traits::Num
    + num_traits::NumRef
    + num_traits::RefNum<Self>
    + num_traits::NumAssignOps
    + num_traits::NumAssign
    + num_traits::NumAssignRef
    + num_traits::Bounded
    + num_traits::ToPrimitive
    + num_traits::FromPrimitive
    + std::iter::Sum
{
    /// Returns half of the value.
    #[must_use]
    fn half(self) -> Self {
        self / (Self::one() + Self::one())
    }
}

/// Blanket implementation of `DistanceValue` for all types that satisfy the trait bounds.
impl<T> DistanceValue for T where
    T: PartialEq
        + PartialOrd
        + Copy
        + Display
        + Debug
        + Default
        + num_traits::Num
        + num_traits::NumRef
        + num_traits::RefNum<Self>
        + num_traits::NumAssignOps
        + num_traits::NumAssign
        + num_traits::NumAssignRef
        + num_traits::Bounded
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + std::iter::Sum
{
}

/// A trait for types that can be used as floating-point distance values in clustering algorithms.
///
/// All types that implement `DistanceValue` should be convertible to a `FloatDistanceValue`, which is required for certain metrics like Euclidean and Cosine
/// distance.
///
/// We provide a blanket implementation for all types that satisfy the trait bounds. This includes all primitive float types.
pub trait FloatDistanceValue: DistanceValue + num_traits::Float + num_traits::FloatConst + num_traits::Pow<Self, Output = Self> {
    /// Converts a `DistanceValue` to a `FloatDistanceValue`.
    #[must_use]
    fn from_distance_value<T: DistanceValue>(value: T) -> Self {
        Self::from(value).unwrap_or_else(|| unreachable!("Failed to convert distance value {value} to float distance value."))
    }
}

impl<T> FloatDistanceValue for T where T: DistanceValue + num_traits::Float + num_traits::FloatConst + num_traits::Pow<Self, Output = Self> {}
