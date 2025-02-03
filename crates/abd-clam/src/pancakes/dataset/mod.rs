//! Datasets for `PanCAKES`.

use distances::Number;
use rayon::prelude::*;

use crate::{
    cakes::{ParSearchable, Searchable},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Dataset, FlatVec, Metric,
};

use super::SquishyBall;

mod codec_data;
mod compression;
mod decompression;

pub use codec_data::CodecData;
pub use compression::{Compressible, Encoder, ParCompressible, ParEncoder};
pub use decompression::{Decoder, Decompressible, ParDecoder, ParDecompressible};

impl<I, Me, Enc: Encoder<I>> Compressible<I, Enc> for FlatVec<I, Me> {}
impl<I: Send + Sync, Me: Send + Sync, Enc: ParEncoder<I>> ParCompressible<I, Enc> for FlatVec<I, Me> {}

impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, Me, Enc: Encoder<I>, Dec: Decoder<I>>
    Searchable<I, T, SquishyBall<T, C>, M> for CodecData<I, Me, Enc, Dec>
{
    fn query_to_center(&self, metric: &M, query: &I, cluster: &SquishyBall<T, C>) -> T {
        metric.distance(query, self.get(cluster.arg_center()))
    }

    fn query_to_all(&self, metric: &M, query: &I, cluster: &SquishyBall<T, C>) -> impl Iterator<Item = (usize, T)> {
        let leaf_bytes = self.leaf_bytes();

        cluster
            .leaves()
            .into_iter()
            .map(SquishyBall::arg_center)
            .map(|o| {
                leaf_bytes
                    .iter()
                    .position(|(off, _)| *off == o)
                    .unwrap_or_else(|| unreachable!("Offset not found in leaf offsets: {o}, {:?}", self.leaf_bytes()))
            })
            .map(|pos| &leaf_bytes[pos])
            .flat_map(|(_, bytes)| self.decode_leaf(bytes, self.decoder()))
            .map(|(i, p)| (i, metric.distance(query, &p)))
    }
}

impl<
        I: Send + Sync,
        T: Number,
        C: ParCluster<T>,
        M: ParMetric<I, T>,
        Me: Send + Sync,
        Enc: ParEncoder<I>,
        Dec: ParDecoder<I>,
    > ParSearchable<I, T, SquishyBall<T, C>, M> for CodecData<I, Me, Enc, Dec>
{
    fn par_query_to_center(&self, metric: &M, query: &I, cluster: &SquishyBall<T, C>) -> T {
        metric.par_distance(query, self.get(cluster.arg_center()))
    }

    fn par_query_to_all(
        &self,
        metric: &M,
        query: &I,
        cluster: &SquishyBall<T, C>,
    ) -> impl rayon::prelude::ParallelIterator<Item = (usize, T)> {
        let leaf_bytes = self.leaf_bytes();

        cluster
            .leaves()
            .into_par_iter()
            .map(SquishyBall::arg_center)
            .map(|o| {
                leaf_bytes
                    .iter()
                    .position(|(off, _)| *off == o)
                    .unwrap_or_else(|| unreachable!("Offset not found in leaf offsets: {o}, {:?}", self.leaf_bytes()))
            })
            .map(|pos| &leaf_bytes[pos])
            .flat_map(|(_, bytes)| self.decode_leaf(bytes, self.decoder()))
            .map(|(i, p)| (i, metric.par_distance(query, &p)))
    }
}
