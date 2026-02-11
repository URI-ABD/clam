//! Compression and compressive search algorithms.
//!
//! This module enables the [`Tree::compress_all`](crate::Tree::compress_all) and [`Tree::decompress_all`](crate::Tree::decompress_all) methods, along with
//! their parallel variants. These methods require that the items stored in the tree implement the [`Codec`] trait, and use that trait implementation to create
//! compressed trees. The compressed trees can be used with the [`CompressiveSearch`] and [`ParCompressiveSearch`] algorithms, which are variants of the
//! [`Search`](crate::cakes::Search) and [`ParSearch`](crate::cakes::ParSearch) algorithms that can perform search in a compressed space while only
//! decompressing those items that are needed to compute the distances to a query.

#[cfg(feature = "serde")]
use std::io::Read;

mod search;
mod tree;

pub use search::{CompressiveSearch, ParCompressiveSearch};

/// Methods for compressing and decompressing items in terms of other items.
///
/// This trait abstracts over the specific compression algorithm used, and allows us to use different compression algorithms for different types of items in the
/// tree. Given two uncompressed items, the `compress` method should return a compressed representation of the target item in terms of the reference item.
/// Conversely, given a reference item and a compressed representation of a target item, the `decompress` method should return the original target item. The
/// `compressed_size` and `original_size` methods should return the number of bytes required to store the compressed representation and the original item,
/// respectively. These are used to determine whether recursive or unitary compression is better for each cluster in the tree.
pub trait Codec {
    /// The type of the compressed representation.
    type Compressed;

    /// Encodes the `target` item in terms of `self`.
    fn compress(&self, target: &Self) -> Self::Compressed;

    /// Decodes the `compressed` representation in terms of `self`.
    #[must_use]
    fn decompress(&self, compressed: &Self::Compressed) -> Self;

    /// Returns the number of bytes in the `compressed` representation.
    fn compressed_size(compressed: &Self::Compressed) -> usize;

    /// Returns the number of bytes in the original representation of `self`.
    fn original_size(&self) -> usize;
}

/// An item that might be stored in a compressed form.
///
/// These are used in the compressed trees, so that we can store some items in their original form and some items in their compressed form, and only decompress
/// the items that we need to compute the distances to the query during [`CompressiveSearch`] and [`ParCompressiveSearch`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub enum MaybeCompressed<I: Codec> {
    /// The original item.
    Original(I),
    /// The compressed representation of the item.
    Compressed(I::Compressed),
}

impl<I: Codec> MaybeCompressed<I> {
    /// Returns the original item if the item is stored in its original form, and None otherwise.
    pub(crate) fn take_original(self) -> Option<I> {
        match self {
            Self::Original(item) => Some(item),
            Self::Compressed(_) => None,
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

    /// Returns the number of bytes required to store the item.
    pub fn size(&self) -> usize {
        match self {
            Self::Original(item) => item.original_size(),
            Self::Compressed(compressed) => I::compressed_size(compressed),
        }
    }

    /// Returns the distance from the query to this item if it is in its original form, and an error otherwise.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to compute the distance to.
    /// * `metric` - The metric to use to compute the distance.
    ///
    /// # Errors
    ///
    /// If the item is in its compressed form.
    pub fn distance_to_query<T, M>(&self, query: &I, metric: &M) -> Result<T, String>
    where
        M: Fn(&I, &I) -> T,
    {
        self.original().map_or_else(
            || Err("Tried to compute distance on a compressed item.".to_string()),
            |item| Ok(metric(item, query)),
        )
    }
}

/// Implementation of [`databuf::Encode`] for [`MaybeCompressed`], gated by the `serde` feature.
#[cfg(feature = "serde")]
impl<I> databuf::Encode for MaybeCompressed<I>
where
    I: Codec + databuf::Encode,
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
    I: Codec + databuf::Decode<'de>,
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
