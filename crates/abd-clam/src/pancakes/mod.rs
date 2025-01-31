//! Compression and Decompression with CLAM

mod cluster;
mod dataset;
// mod sequence;

pub use cluster::{SquishCosts, SquishyBall};
pub use dataset::{
    CodecData, Compressible, Decoder, Decompressible, Encoder, ParCompressible, ParDecoder, ParDecompressible,
    ParEncoder,
};
