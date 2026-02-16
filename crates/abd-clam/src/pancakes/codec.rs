//! Traits and structs for compression and decompression in Pancakes.

#[cfg(feature = "serde")]
use std::io::Read;

use crate::DistanceValue;

/// An item that can be compressed and decompressed in terms of another item, and whose size, in either form, can be computed.
pub trait Compressible {
    /// The compressed representation of the item.
    type Compressed;

    /// Returns the size of the original item.
    fn original_size(&self) -> usize;

    /// Returns the size of the compressed representation of the item.
    fn compressed_size(compressed: &Self::Compressed) -> usize;
}

/// A object that can compress and decompress items that implement [`Compressible`].
pub trait Codec<I: Compressible> {
    /// Compresses the `target` in terms of the `reference`, returning the compressed representation of the `target`.
    fn compress(&self, reference: &I, target: &I) -> I::Compressed;

    /// Decompresses the `compressed` representation of the item using the `reference`, returning the original item.
    fn decompress(&self, reference: &I, compressed: &I::Compressed) -> I;
}

#[cfg(feature = "musals")]
impl Compressible for crate::musals::AlignedSequence {
    type Compressed = crate::musals::Edits;

    fn original_size(&self) -> usize {
        deepsize::DeepSizeOf::deep_size_of(self)
    }

    fn compressed_size(compressed: &Self::Compressed) -> usize {
        deepsize::DeepSizeOf::deep_size_of(compressed)
    }
}

#[cfg(feature = "musals")]
impl<T: DistanceValue> Codec<crate::musals::AlignedSequence> for crate::musals::CostMatrix<T> {
    fn compress(&self, center: &crate::musals::AlignedSequence, other: &crate::musals::AlignedSequence) -> crate::musals::Edits {
        let [edits, _] = center.nw_edit_scripts(other, self);
        edits
    }

    fn decompress(&self, center: &crate::musals::AlignedSequence, edits: &crate::musals::Edits) -> crate::musals::AlignedSequence {
        center.clone().apply_edits(edits)
    }
}

/// An item that might be stored in a compressed form or in its original form in the tree.
///
/// These are used in the compressed (or partially compressed) trees, so that we can store some items in their original form and some items in their compressed
/// form, and only decompress the items that we need to compute the distances to the query during [`CompressiveSearch`](super::CompressiveSearch).
#[must_use]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MaybeCompressed<I: Compressible> {
    /// The original item.
    Original(I),
    /// The compressed representation of the item.
    Compressed(I::Compressed),
}

impl<I: Compressible + deepsize::DeepSizeOf> deepsize::DeepSizeOf for MaybeCompressed<I> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        match self {
            Self::Original(item) => item.deep_size_of_children(context),
            Self::Compressed(compressed) => I::compressed_size(compressed),
        }
    }
}

impl<I: Compressible> MaybeCompressed<I> {
    /// Returns the original item if the item is stored in its original form, and None otherwise.
    pub fn take_original(self) -> Option<I> {
        match self {
            Self::Original(item) => Some(item),
            Self::Compressed(_) => None,
        }
    }

    /// Returns the compressed representation of the item if the item is stored in its compressed form, and None otherwise.
    pub fn take_compressed(self) -> Option<I::Compressed> {
        match self {
            Self::Original(_) => None,
            Self::Compressed(compressed) => Some(compressed),
        }
    }

    /// Returns a reference to the original item if the item is stored in its original form, and None otherwise.
    pub const fn original(&self) -> Option<&I> {
        match self {
            Self::Original(item) => Some(item),
            Self::Compressed(_) => None,
        }
    }

    /// Returns a reference to the compressed representation of the item if the item is stored in its compressed form, and None otherwise.
    pub const fn compressed(&self) -> Option<&I::Compressed> {
        match self {
            Self::Original(_) => None,
            Self::Compressed(compressed) => Some(compressed),
        }
    }

    /// Returns the size of the original item if the item is stored in its original form, and `None` otherwise.
    pub fn original_size(&self) -> Option<usize> {
        match self {
            Self::Original(item) => Some(item.original_size()),
            Self::Compressed(_) => None,
        }
    }

    /// Returns the size of the compressed representation if the item is stored in its compressed form, and `None` otherwise.
    pub fn compressed_size(&self) -> Option<usize> {
        match self {
            Self::Original(_) => None,
            Self::Compressed(compressed) => Some(I::compressed_size(compressed)),
        }
    }

    /// Returns the size of the item in whichever form it is currently stored (compressed or original).
    pub fn size(&self) -> usize {
        match self {
            Self::Original(item) => item.original_size(),
            Self::Compressed(compressed) => I::compressed_size(compressed),
        }
    }

    /// If this item is stored in its original form, returns the distance to the other (uncompressed) item using the provided metric.
    ///
    /// # Arguments
    ///
    /// * `uncompressed` - The uncompressed item to compute the distance to.
    /// * `metric` - The metric to use to compute the distance.
    ///
    /// # Errors
    ///
    /// If the item is in its compressed form.
    pub fn distance_to_uncompressed<T, M>(&self, uncompressed: &I, metric: &M) -> Result<T, String>
    where
        M: Fn(&I, &I) -> T,
    {
        self.original().map_or_else(
            || Err("Tried to compute distance on a compressed item.".to_string()),
            |item| Ok(metric(item, uncompressed)),
        )
    }

    /// If this item and the other item are both stored in their original form, returns the distance between them using the provided metric.
    ///
    /// # Arguments
    ///
    /// * `other` - The other item to compute the distance to.
    /// * `metric` - The metric to use to compute the distance.
    ///
    /// # Errors
    ///
    /// If either item is in its compressed form.
    pub fn distance_to_other<T, M>(&self, other: &Self, metric: &M) -> Result<T, String>
    where
        M: Fn(&I, &I) -> T,
    {
        match (self.original(), other.original()) {
            (Some(item1), Some(item2)) => Ok(metric(item1, item2)),
            _ => Err("Tried to compute distance on a compressed item.".to_string()),
        }
    }
}

/// Implementation of [`databuf::Encode`] for [`MaybeCompressed`], gated by the `serde` feature.
#[cfg(feature = "serde")]
impl<I> databuf::Encode for MaybeCompressed<I>
where
    I: Compressible + databuf::Encode,
    I::Compressed: databuf::Encode,
{
    fn encode<const CONFIG: u16>(&self, buf: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::Original(item) => {
                buf.write_all(&[0])?;
                item.encode::<CONFIG>(buf)
            }
            Self::Compressed(compressed) => {
                buf.write_all(&[1])?;
                compressed.encode::<CONFIG>(buf)
            }
        }
    }
}

/// Implementation of [`databuf::Decode`] for [`MaybeCompressed`], gated by the `serde` feature.
#[cfg(feature = "serde")]
impl<'de, I> databuf::Decode<'de> for MaybeCompressed<I>
where
    I: Compressible + databuf::Decode<'de>,
    I::Compressed: databuf::Decode<'de>,
{
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let mut variant = [0];
        buffer.read_exact(&mut variant)?;
        match variant[0] {
            0 => Ok(Self::Original(I::decode::<CONFIG>(buffer)?)),
            1 => Ok(Self::Compressed(I::Compressed::decode::<CONFIG>(buffer)?)),
            _ => Err("Invalid variant for MaybeCompressed".to_string().into()),
        }
    }
}
