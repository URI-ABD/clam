//! Compression and Decompression

mod codec;
pub mod knn;
pub mod rnn;
mod search;

pub use codec::{decode_general, encode_general, CodecData, DecoderFn, EncoderFn, SquishyBall};
