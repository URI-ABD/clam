//! A variant of `FlatVec` that is used to store the data in a mass-spring
//! system for dimensionality reduction.

mod spring;
mod system;
mod vector;

pub use spring::Spring;
pub use system::System;
pub use vector::Vector;
