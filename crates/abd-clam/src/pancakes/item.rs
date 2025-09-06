//! An item in a dataset, which can be either encoded or decoded.

use super::{Decoder, Encoder, ParDecoder, ParEncoder};

/// The item in a dataset, which can be either encoded or decoded.
#[derive(Debug)]
#[must_use]
pub enum CodecItem<I, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> {
    /// The item is encoded in terms of item of a reference item.
    Encoded(Enc::Bytes),
    /// The item is decoded and available for use.
    Decoded(I),
}

impl<I, Enc, Dec> Clone for CodecItem<I, Enc, Dec>
where
    I: Clone,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn clone(&self) -> Self {
        match self {
            Self::Encoded(encoded) => Self::Encoded(encoded.clone()),
            Self::Decoded(item) => Self::Decoded(item.clone()),
        }
    }
}

impl<I, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> CodecItem<I, Enc, Dec> {
    /// Creates a new encoded item.
    pub const fn new_encoded(encoded: Enc::Bytes) -> Self {
        Self::Encoded(encoded)
    }

    /// Creates a new decoded item.
    pub const fn new_decoded(item: I) -> Self {
        Self::Decoded(item)
    }

    /// Encodes the item if it is decoded, using the provided encoder and
    /// reference item.
    pub fn encode(mut self, encoder: &Enc, reference: &I) -> Self {
        match self {
            Self::Decoded(item) => {
                self = Self::Encoded(encoder.encode(&item, reference));
            }
            Self::Encoded(_) => {
                // Nothing to do if already encoded.
            }
        }
        self
    }

    /// Decodes the item if it is encoded, using the provided decoder and
    /// reference item.
    ///
    /// # Errors
    ///
    ///   - If decoding fails, returns the error from the decoder.
    pub fn decode(mut self, decoder: &Dec, reference: &I) -> Result<Self, Dec::Err> {
        match self {
            Self::Decoded(_) => (),
            Self::Encoded(encoded) => {
                let item = decoder.decode(reference, &encoded)?;
                self = Self::Decoded(item);
            }
        }
        Ok(self)
    }
}

impl<I: Send + Sync, Enc: ParEncoder<I, Dec>, Dec: ParDecoder<I, Enc>> CodecItem<I, Enc, Dec>
where
    Enc::Bytes: Send + Sync,
    Dec::Err: Send + Sync,
{
    /// Parallel version of [`encode`](Self::encode).
    pub fn par_encode(mut self, encoder: &Enc, reference: &I) -> Self {
        match self {
            Self::Decoded(item) => {
                self = Self::Encoded(encoder.par_encode(&item, reference));
            }
            Self::Encoded(_) => {
                // Nothing to do if already encoded.
            }
        }
        self
    }

    /// Parallel version of [`decode`](Self::decode).
    ///
    /// # Errors
    ///
    /// See [`decode`](Self::decode) for details.
    pub fn par_decode(mut self, decoder: &Dec, reference: &I) -> Result<Self, Dec::Err> {
        match self {
            Self::Decoded(_) => (),
            Self::Encoded(encoded) => {
                let item = decoder.par_decode(reference, &encoded)?;
                self = Self::Decoded(item);
            }
        }
        Ok(self)
    }
}
