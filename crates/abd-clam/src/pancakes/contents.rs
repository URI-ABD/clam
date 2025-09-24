//! The contents of a `SquishedBall`, which can be either recursively encoded
//! (i.e., containing child clusters) or a leaf cluster (i.e., containing items
//! encoded as deltas against the center of the cluster).

use crate::DistanceValue;

use super::{CodecItem, Decoder, Encoder, SquishedBall};

/// The contents of a `SquishedBall` are either the children clusters if it was
/// recursively encoded, or the encoded items if it is a leaf cluster.
#[must_use]
pub enum CodecContents<I, T: DistanceValue, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> {
    /// The cluster was recursively encoded, and contains children clusters.
    Recursive(Vec<Box<SquishedBall<I, T, Enc, Dec>>>),
    /// The cluster is a leaf, and contains items.
    Leaf(Vec<CodecItem<I, Enc, Dec>>),
}

impl<I, T: DistanceValue, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> CodecContents<I, T, Enc, Dec> {
    /// Encodes all items in the subtree rooted at this node, using the provided
    /// encoder and the center of the node as the reference item.
    pub fn encode_subtree(&mut self, encoder: &Enc, center: &I) {
        match self {
            Self::Recursive(children) => {
                *children = children
                    .drain(..)
                    .map(|mut child| {
                        child.encode_subtree(encoder, center);
                        child
                    })
                    .collect();
            }
            Self::Leaf(items) => {
                *items = items
                    .drain(..)
                    .map(|item| item.encoded(encoder, center))
                    .map(CodecItem::Encoded)
                    .collect();
            }
        }
    }

    /// Decodes all items in the subtree rooted at this node, using the provided
    /// decoder and the center of the parent node as the reference item.
    pub fn decode_subtree(&mut self, decoder: &Dec, center: &I) {
        match self {
            Self::Recursive(children) => {
                *children = children
                    .drain(..)
                    .map(|mut child| {
                        child.decode_subtree(decoder, center);
                        child
                    })
                    .collect();
            }
            Self::Leaf(items) => {
                *items = items
                    .drain(..)
                    .map(|item| item.decoded(decoder, center))
                    .map(CodecItem::Decoded)
                    .collect();
            }
        }
    }

    /// Decodes all items in the subtree rooted at this node, using the provided
    /// decoder and the center of the parent node as the reference item, and
    /// returns them as a flat vector.
    pub fn decode_all(self, decoder: &Dec, center: &I) -> Vec<I> {
        match self {
            Self::Recursive(children) => children
                .into_iter()
                .flat_map(|child| child.contents.decode_all(decoder, center))
                .collect(),
            Self::Leaf(items) => items
                .into_iter()
                .map(|item| match item {
                    CodecItem::Decoded(item) => item,
                    CodecItem::Encoded(encoded) => decoder.decode(center, &encoded),
                })
                .collect(),
        }
    }
}
