//! The `Number` trait is used to represent numbers of different types.
//!
//! We provide implementations for the following types:
//!
//! * All primitive unsigned integers: `u8`, `u16`, `u32`, `u64`, `u128`, `usize`.
//! * All primitive signed integers: `i8`, `i16`, `i32`, `i64`, `i128`, `isize`.
//! * All primitive floating point numbers: `f32`, `f64`.

mod _bool;
mod _number;
mod _variants;

pub use _bool::Bool;
pub use _number::Number;
pub use _variants::{Float, IInt, Int, UInt};
