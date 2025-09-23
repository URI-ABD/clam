//! An item in a dataset, which can be either encoded or decoded.

use super::{Decoder, Encoder};

/// The item in a dataset, which can be either encoded or decoded.
#[must_use]
pub enum CodecItem<I, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> {
    /// The item is encoded in terms of item of a reference item.
    Encoded(Enc::Output),
    /// The item is decoded and available for use.
    Decoded(I),
}

impl<I, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> CodecItem<I, Enc, Dec> {
    /// Encodes the item if it is decoded, using the provided encoder and
    /// reference item.
    pub fn encoded(self, encoder: &Enc, reference: &I) -> Enc::Output {
        match self {
            Self::Decoded(item) => encoder.encode(&item, reference),
            Self::Encoded(delta) => delta,
        }
    }

    /// Decodes the item if it is encoded, using the provided decoder and
    /// reference item.
    pub fn decoded(self, decoder: &Dec, reference: &I) -> I {
        match self {
            Self::Decoded(item) => item,
            Self::Encoded(delta) => decoder.decode(reference, &delta),
        }
    }
}
