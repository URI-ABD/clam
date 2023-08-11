#![allow(
    //clippy::unwrap_used,
    clippy::module_name_repetitions
)]

pub use _constructable::ConstructableNumber;
pub use dataset::BatchedArrowDataset;

/// The main user-facing dataset implementation
mod dataset;

/// IPC metadata information and parsing
mod metadata;

/// IPC batch reader. The glue between individual arrow files
mod reader;

/// Various file i/o helpers and utilities
mod io;

/// Tests
mod tests;

/// File generation utility functionality
mod util;

/// Constructable number trait (TBR)
mod _constructable;
