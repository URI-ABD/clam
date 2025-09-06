//! The contents of a `SquishedBall`, which can be either recursively encoded
//! (i.e., containing child clusters) or a leaf cluster (i.e., containing items
//! encoded as deltas against the center of the cluster).

use rayon::prelude::*;

use crate::{Cluster, DistanceValue};

use super::{CodecItem, Decoder, Encoder, ParDecoder, ParEncoder, SquishedBall};

/// The contents of a `SquishedBall` are either the children clusters if it was
/// recursively encoded, or the encoded items if it is a leaf cluster.
#[must_use]
pub enum CodecContents<I, T: DistanceValue, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> {
    /// The cluster was recursively encoded, and contains children clusters.
    Recursive(Vec<Box<SquishedBall<I, T, Enc, Dec>>>),
    /// The cluster is a leaf, and contains items.
    Leaf(Vec<CodecItem<I, Enc, Dec>>),
}

impl<I, T, Enc, Dec> Clone for CodecContents<I, T, Enc, Dec>
where
    I: Clone,
    T: DistanceValue + Clone,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn clone(&self) -> Self {
        match self {
            Self::Recursive(children) => Self::Recursive(children.clone()),
            Self::Leaf(items) => Self::Leaf(items.clone()),
        }
    }
}

impl<I, T: DistanceValue, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> CodecContents<I, T, Enc, Dec> {
    /// Decodes all items in the subtree rooted at this node, using the provided
    /// decoder and the center of the parent node as the reference item.
    ///
    /// # Errors
    ///
    /// If any item fails to decode.
    pub fn decode_subtree(&mut self, decoder: &Dec, parent_center: &I) -> Result<(), Dec::Err> {
        match self {
            Self::Recursive(children) => {
                *children = children
                    .drain(..)
                    .map(|mut child| {
                        match &child.center {
                            CodecItem::Encoded(encoded) => {
                                let center = decoder.decode(parent_center, encoded)?;
                                child.contents.decode_subtree(decoder, &center)?;
                                child.center = CodecItem::Decoded(center);
                            }
                            CodecItem::Decoded(center) => {
                                child.contents.decode_subtree(decoder, center)?;
                            }
                        }
                        Ok(child)
                    })
                    .collect::<Result<_, _>>()?;
            }
            Self::Leaf(items) => {
                *items = items
                    .drain(..)
                    .map(|item| item.decode(decoder, parent_center))
                    .collect::<Result<_, _>>()?;
            }
        }
        Ok(())
    }

    /// Decodes all items in the subtree rooted at this node, using the provided
    /// decoder and the center of the parent node as the reference item, and
    /// returns them as a flat vector.
    ///
    /// # Errors
    ///
    /// If any item fails to decode.
    pub fn decode_all(self, decoder: &Dec, parent_center: &I) -> Result<Vec<I>, Dec::Err> {
        match self {
            Self::Recursive(children) => {
                let mut items = Vec::with_capacity(children.iter().map(|c| c.cardinality()).sum());

                for mut child in children {
                    match &child.center {
                        CodecItem::Decoded(center) => {
                            items.extend(child.contents.decode_all(decoder, center)?);
                        }
                        CodecItem::Encoded(encoded) => {
                            let center = decoder.decode(parent_center, encoded)?;
                            items.extend(child.contents.decode_all(decoder, &center)?);
                            child.center = CodecItem::Decoded(center);
                        }
                    }
                }

                Ok(items)
            }
            Self::Leaf(items) => items
                .into_iter()
                .map(|item| match item {
                    CodecItem::Decoded(item) => Ok(item),
                    CodecItem::Encoded(encoded) => decoder.decode(parent_center, &encoded),
                })
                .collect(),
        }
    }
}

impl<I, T, Enc, Dec> CodecContents<I, T, Enc, Dec>
where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    Enc: ParEncoder<I, Dec>,
    Dec: ParDecoder<I, Enc>,
    Enc::Bytes: Send + Sync,
    Dec::Err: Send + Sync,
{
    /// Parallel version of [`decode_subtree`](Self::decode_subtree).
    ///
    /// # Errors
    ///
    /// See [`decode_subtree`](Self::decode_subtree).
    pub fn par_decode_subtree(&mut self, decoder: &Dec, parent_center: &I) -> Result<(), Dec::Err> {
        match self {
            Self::Recursive(children) => {
                *children = children
                    .par_drain(..)
                    .map(|mut child| {
                        match &child.center {
                            CodecItem::Encoded(encoded) => {
                                let center = decoder.par_decode(parent_center, encoded)?;
                                child.contents.par_decode_subtree(decoder, &center)?;
                                child.center = CodecItem::Decoded(center);
                            }
                            CodecItem::Decoded(center) => {
                                child.contents.par_decode_subtree(decoder, center)?;
                            }
                        }
                        Ok(child)
                    })
                    .collect::<Result<_, _>>()?;
            }
            Self::Leaf(items) => {
                *items = items
                    .par_drain(..)
                    .map(|item| item.par_decode(decoder, parent_center))
                    .collect::<Result<_, _>>()?;
            }
        }
        Ok(())
    }

    /// Parallel version of [`decode_all`](Self::decode_all).
    ///
    /// # Errors
    ///
    /// See [`decode_all`](Self::decode_all).
    pub fn par_decode_all(self, decoder: &Dec, parent_center: &I) -> Result<Vec<I>, Dec::Err> {
        match self {
            Self::Recursive(mut children) => {
                let items = children
                    .par_drain(..)
                    .flat_map(|mut child| match &child.center {
                        CodecItem::Decoded(center) => child.contents.par_decode_all(decoder, center),
                        CodecItem::Encoded(encoded) => {
                            let center = decoder.decode(parent_center, encoded)?;
                            let items = child.contents.par_decode_all(decoder, &center)?;
                            child.center = CodecItem::Decoded(center);
                            Ok(items)
                        }
                    })
                    .flatten()
                    .collect::<Vec<_>>();
                Ok(items)
            }
            Self::Leaf(items) => items
                .into_par_iter()
                .map(|item| match item {
                    CodecItem::Decoded(item) => Ok(item),
                    CodecItem::Encoded(encoded) => decoder.par_decode(parent_center, &encoded),
                })
                .collect(),
        }
    }
}
