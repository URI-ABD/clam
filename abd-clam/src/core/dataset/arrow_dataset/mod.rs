#![allow(
    clippy::missing_const_for_fn,
    clippy::uninlined_format_args,
    clippy::redundant_clone,
    clippy::default_trait_access,
    clippy::use_self,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::type_complexity,
    clippy::option_if_let_else,
    clippy::inconsistent_struct_constructor,
    clippy::redundant_pub_crate,
    clippy::used_underscore_binding,
    clippy::missing_docs_in_private_items,
    clippy::module_name_repetitions,
    clippy::unwrap_used,
    clippy::expect_used,
    dead_code
)]

// The main user-facing dataset implementation
pub use _constructable::ConstructableNumber;
pub use dataset::BatchedArrowDataset;
mod dataset;

// IPC metadata information and parsing
mod metadata;

// IPC batch reader. The glue between individual arrow files
mod reader;

// Various file i/o helpers and utilities
mod io;

mod tests;
mod util;

mod _constructable;
