//! Traits and types for use in compression and decompression with `PanCAKES`.

mod contents;
mod item;
mod squished_ball;
mod squishy_ball;

pub use squished_ball::SquishedBall;

use contents::CodecContents;
use item::CodecItem;
use squishy_ball::SquishyBall;

/// A trait for objects that can be used to encode items of type `T`.
///
/// The items can be encoded by themselves or as deltas against a reference
/// item. The encoded items are used for compression with the assumtion that
/// the memory cost of the encoded representation decreases as the similarity
/// between the item and the reference increases.
pub trait Encoder<T, Dec: Decoder<T, Self>>: Sized {
    /// The type of representation used by this encoder.
    type Output;

    /// Encode an item by itself without using a reference item.
    fn encode_raw(&self, item: &T) -> Self::Output;

    /// Encode an item as a delta against a reference item.
    fn encode(&self, item: &T, reference: &T) -> Self::Output;

    /// Get an estimate of the length of a delta without actually encoding it.
    fn estimate_delta_size(&self, item: &T, reference: &T) -> usize;
}

/// A trait for objects that can be used to decode items of type `T`.
///
/// The items can be decoded from bytes and from deltas against a reference
/// item.
pub trait Decoder<T, Enc: Encoder<T, Self>>: Sized {
    /// Decode an item from its byte representation.
    fn decode_raw(&self, bytes: &Enc::Output) -> T;

    /// Decode an item from a delta against a reference item.
    fn decode(&self, reference: &T, delta: &Enc::Output) -> T;
}

/// A macro to implement `Encoder` and `Decoder` for primitive distance value types.
macro_rules! impl_codec_for_dist_val {
    ($($T:ty),*) => {
        $(
            impl Encoder<$T, ()> for () {
                type Output = [u8; core::mem::size_of::<$T>()];

                fn encode_raw(&self, item: &$T) -> Self::Output {
                    item.to_be_bytes()
                }

                fn encode(&self, item: &$T, _: &$T) -> Self::Output {
                    self.encode_raw(item)
                }

                fn estimate_delta_size(&self, _: &$T, _: &$T) -> usize {
                    core::mem::size_of::<$T>()
                }
            }

            impl Decoder<$T, ()> for () {
                fn decode_raw(&self, bytes: &<() as Encoder<$T, ()>>::Output) -> $T {
                    <$T>::from_be_bytes(*bytes)
                }

                fn decode(&self, _: &$T, delta: &[u8; core::mem::size_of::<$T>()]) -> $T {
                    self.decode_raw(delta)
                }
            }
        )*
    };
}

impl_codec_for_dist_val!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);

/// Parallel version of [`Encoder`](Encoder).
pub trait ParEncoder<I, Dec: ParDecoder<I, Self>>: Send + Sync + Encoder<I, Dec>
where
    Self::Output: Send + Sync,
{
    /// Parallel version of [`encode_raw`](Encoder::encode_raw).
    fn par_encode_raw(&self, item: &I) -> Self::Output;

    /// Parallel version of [`encode`](Encoder::encode).
    fn par_encode(&self, item: &I, reference: &I) -> Self::Output;

    /// Parallel version of [`estimate_delta_size`](Encoder::estimate_delta_size).
    fn par_estimate_delta_size(&self, item: &I, reference: &I) -> usize;
}

/// Parallel version of [`Decoder`](Decoder).
pub trait ParDecoder<I, Enc: ParEncoder<I, Self>>: Send + Sync + Decoder<I, Enc>
where
    Enc::Output: Send + Sync,
{
    /// Parallel version of [`decode_raw`](Decoder::decode_raw).
    fn par_decode_raw(&self, bytes: &Enc::Output) -> I;

    /// Parallel version of [`decode`](Decoder::decode).
    fn par_decode(&self, reference: &I, delta: &Enc::Output) -> I;
}

/// Parallel versions of the [`impl_codec_for_dist_val`](impl_codec_for_dist_val) macro.
macro_rules! impl_par_codec_for_dist_val {
    ($($T:ty),*) => {
        $(
            impl ParEncoder<$T, ()> for () {
                fn par_encode_raw(&self, item: &$T) -> Self::Output {
                    item.to_be_bytes()
                }

                fn par_encode(&self, item: &$T, _: &$T) -> Self::Output {
                    self.par_encode_raw(item)
                }

                fn par_estimate_delta_size(&self, _: &$T, _: &$T) -> usize {
                    core::mem::size_of::<$T>()
                }
            }

            impl ParDecoder<$T, ()> for () {
                fn par_decode_raw(&self, bytes: &<() as Encoder<$T, ()>>::Output) -> $T {
                    <$T>::from_be_bytes(*bytes)
                }

                fn par_decode(&self, _: &$T, delta: &[u8; core::mem::size_of::<$T>()]) -> $T {
                    self.par_decode_raw(delta)
                }
            }
        )*
    };
}

impl_par_codec_for_dist_val!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);
