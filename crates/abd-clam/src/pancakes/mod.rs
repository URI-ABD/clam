//! Compression and Decompression

mod codec;
mod knn;
mod rnn;
mod search;

pub use codec::{decode_general, encode_general, CodecData, DecoderFn, EncoderFn, SquishyBall};
