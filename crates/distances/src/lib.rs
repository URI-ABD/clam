#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]
// #![no_std]  // TODO: re-enable as an optional feature

// extern crate alloc;

pub mod number;

pub use number::Number;

pub mod sets;
pub mod simd;
pub mod strings;
pub mod vectors;

/// The version of the crate.
pub const VERSION: &str = "1.6.3";
