#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items
)]
#![doc = include_str!("../README.md")]

pub mod augmentation;
pub mod random_data;
pub mod random_edits;

/// The version of the crate.
pub const VERSION: &str = "0.4.0";
