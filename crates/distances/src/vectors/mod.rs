//! Distance functions for vectors.
//!
//! # Potentially unexpected behaviors
//! Computing many of these distances with vectors of differing or zero
//! dimensionality may give unexpected results. Specifically, when one vector is
//! shorter than the other, elements in the longer vector past the end of the
//! shorter vector will be ignored.

mod angular;
mod lp_norms;
pub(crate) mod utils;

pub use angular::{bray_curtis, canberra, cosine, hamming};
pub use lp_norms::{
    chebyshev, euclidean, euclidean_sq, l3_norm, l4_norm, manhattan, minkowski, minkowski_p,
};
