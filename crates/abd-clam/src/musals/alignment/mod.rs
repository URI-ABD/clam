//! Helper types and functions for Musals.

mod alignment_ops;
mod cost_matrix;
mod needleman_wunsch;

pub use alignment_ops::{Direction, Edit, Edits, Substitutions};
pub use cost_matrix::CostMatrix;
