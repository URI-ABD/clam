//! An extension trait for `Dataset` that provides encoding and decoding methods for metrics.

use distances::Number;

use super::{Dataset, Instance};

/// An extension trait for `Dataset` that provides encoding and decoding methods for metrics.
pub trait Codec<I: Instance, U: Number>: Dataset<I, U> {
    /// Encodes an instance in the dataset into a byte array using a reference instance.
    fn encode_instance(&self, reference: &I, target: &I) -> Box<[u8]>;

    /// Decodes an instance from a byte array using a reference instance.
    fn decode_instance(&self, reference: &I, encoding: &[u8]) -> I;
}
