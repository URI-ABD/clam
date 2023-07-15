//! Distance functions for vectors.

mod angular;
mod lp_norms;
pub(crate) mod utils;

pub use angular::{canberra, cosine, hamming};
pub use lp_norms::{
    chebyshev, euclidean, euclidean_sq, l3_norm, l4_norm, manhattan, minkowski, minkowski_p,
};
