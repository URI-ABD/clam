//! The data types of the instances in the datasets.

mod aligned_sequence;
mod member_set;
mod unaligned_sequence;

pub use aligned_sequence::{Aligned, Hamming};
pub use member_set::{Jaccard, MemberSet};
pub use unaligned_sequence::Unaligned;
