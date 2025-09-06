#![doc = include_str!("../README.md")]
// #![no_std]  // TODO Najib: re-enable as an optional feature

// extern crate alloc;

pub mod number;

pub use number::Number;

pub mod blas;
pub mod sets;
pub mod simd;
pub mod strings;
pub mod vectors;

/// The version of the crate.
pub const VERSION: &str = "1.8.0";
