//! Traits and types for use in compression and decompression with `PanCAKES`.

use crate::DistanceValue;

mod contents;
mod item;
mod squished_ball;
mod squishy_ball;

use contents::CodecContents;
use item::CodecItem;
pub use squished_ball::SquishedBall;
use squishy_ball::SquishyBall;

/// A trait for objects that can be used to encode items of type `T`.
///
/// The items can be encoded to bytes and as deltas against a reference item.
///
/// The default implementation of the [`estimate_delta_size`](Encoder::estimate_delta_size)
/// method uses the distance between the two items as a proxy for the length of
/// the delta. This is a reasonable assumption for many use cases (e.g., *-omic
/// sequences under edit-distance), but may not hold in all cases (e.g. floating
/// point vectors under Euclidean distance). In such cases, the user should
/// override the default implementation of this method.
pub trait Encoder<I, Dec: Decoder<I, Self>>: Sized {
    /// The type of byte representation used by this encoder.
    type Bytes;

    /// Encode an item by itself without using a reference item.
    fn encode_raw(&self, item: &I) -> Self::Bytes;

    /// Encode an item as a delta against a reference item.
    fn encode(&self, item: &I, reference: &I) -> Self::Bytes;

    /// Get an estimate of the length of a delta without actually encoding it.
    fn estimate_delta_size<T: DistanceValue, M: Fn(&I, &I) -> T>(&self, item: &I, reference: &I, metric: &M) -> usize {
        (metric(item, reference))
            .to_usize()
            .unwrap_or(core::mem::size_of::<I>())
    }
}

/// A trait for objects that can be used to decode items of type `T`.
///
/// The items can be decoded from bytes and from deltas against a reference
/// item.
pub trait Decoder<I, Enc: Encoder<I, Self>>: Sized {
    /// Decode an item from its byte representation.
    fn decode_raw(&self, bytes: &Enc::Bytes) -> I;

    /// Decode an item from a delta against a reference item.
    fn decode(&self, reference: &I, delta: &Enc::Bytes) -> I;
}

/// A macro to implement `Encoder` and `Decoder` for primitive distance value types.
macro_rules! impl_codec_for_dist_val {
    ($T: ty, $L: expr) => {
        impl Encoder<$T, ()> for () {
            type Bytes = [u8; $L];

            fn encode_raw(&self, item: &$T) -> Self::Bytes {
                item.to_be_bytes()
            }

            fn encode(&self, item: &$T, _: &$T) -> Self::Bytes {
                self.encode_raw(item)
            }

            fn estimate_delta_size<T_: DistanceValue, M: Fn(&$T, &$T) -> T_>(&self, _: &$T, _: &$T, _: &M) -> usize {
                core::mem::size_of::<$T>()
            }
        }

        impl Decoder<$T, ()> for () {
            fn decode_raw(&self, bytes: &<() as Encoder<$T, ()>>::Bytes) -> $T {
                <$T>::from_be_bytes(*bytes)
            }

            fn decode(&self, _: &$T, delta: &[u8; $L]) -> $T {
                self.decode_raw(delta)
            }
        }
    };
}

impl_codec_for_dist_val!(u8, 1);
impl_codec_for_dist_val!(u16, 2);
impl_codec_for_dist_val!(u32, 4);
impl_codec_for_dist_val!(u64, 8);
impl_codec_for_dist_val!(i8, 1);
impl_codec_for_dist_val!(i16, 2);
impl_codec_for_dist_val!(i32, 4);
impl_codec_for_dist_val!(i64, 8);
impl_codec_for_dist_val!(f32, 4);
impl_codec_for_dist_val!(f64, 8);
impl_codec_for_dist_val!(usize, core::mem::size_of::<usize>());
impl_codec_for_dist_val!(isize, core::mem::size_of::<isize>());

/// Parallel version of [`Encoder`](Encoder).
pub trait ParEncoder<I, Dec: ParDecoder<I, Self>>: Send + Sync + Encoder<I, Dec>
where
    Self::Bytes: Send + Sync,
{
    /// Parallel version of [`encode_raw`](Encoder::encode_raw).
    fn par_encode_raw(&self, item: &I) -> Self::Bytes;

    /// Parallel version of [`encode`](Encoder::encode).
    fn par_encode(&self, item: &I, reference: &I) -> Self::Bytes;

    /// Parallel version of [`estimate_delta_size`](Encoder::estimate_delta_size).
    fn par_estimate_delta_size<T: DistanceValue + Send + Sync, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &self,
        item: &I,
        reference: &I,
        metric: &M,
    ) -> usize {
        (metric(item, reference))
            .to_usize()
            .unwrap_or(core::mem::size_of::<I>())
    }
}

/// Parallel version of [`Decoder`](Decoder).
pub trait ParDecoder<I, Enc: ParEncoder<I, Self>>: Send + Sync + Decoder<I, Enc>
where
    Enc::Bytes: Send + Sync,
{
    /// Parallel version of [`decode_raw`](Decoder::decode_raw).
    fn par_decode_raw(&self, bytes: &Enc::Bytes) -> I;

    /// Parallel version of [`decode`](Decoder::decode).
    fn par_decode(&self, reference: &I, delta: &Enc::Bytes) -> I;
}

/// Parallel versions of the [`impl_codec_for_dist_val`](impl_codec_for_dist_val) macro.
macro_rules! impl_par_codec_for_dist_val {
    ($T: ty, $L: expr) => {
        impl ParEncoder<$T, ()> for () {
            fn par_encode_raw(&self, item: &$T) -> Self::Bytes {
                item.to_be_bytes()
            }

            fn par_encode(&self, item: &$T, _: &$T) -> Self::Bytes {
                self.par_encode_raw(item)
            }

            fn par_estimate_delta_size<T_: DistanceValue, M: Fn(&$T, &$T) -> T_>(&self, _: &$T, _: &$T, _: &M) -> usize {
                core::mem::size_of::<$T>()
            }
        }

        impl ParDecoder<$T, ()> for () {
            fn par_decode_raw(&self, bytes: &<() as Encoder<$T, ()>>::Bytes) -> $T {
                <$T>::from_be_bytes(*bytes)
            }

            fn par_decode(&self, _: &$T, delta: &[u8; $L]) -> $T {
                self.par_decode_raw(delta)
            }
        }
    };
}

impl_par_codec_for_dist_val!(u8, 1);
impl_par_codec_for_dist_val!(u16, 2);
impl_par_codec_for_dist_val!(u32, 4);
impl_par_codec_for_dist_val!(u64, 8);
impl_par_codec_for_dist_val!(i8, 1);
impl_par_codec_for_dist_val!(i16, 2);
impl_par_codec_for_dist_val!(i32, 4);
impl_par_codec_for_dist_val!(i64, 8);
impl_par_codec_for_dist_val!(f32, 4);
impl_par_codec_for_dist_val!(f64, 8);
impl_par_codec_for_dist_val!(usize, core::mem::size_of::<usize>());
impl_par_codec_for_dist_val!(isize, core::mem::size_of::<isize>());
