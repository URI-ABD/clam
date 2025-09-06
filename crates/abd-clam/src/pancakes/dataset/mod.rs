//! Datasets for `PanCAKES`.

mod codec_data;
mod compression;
mod decompression;

pub use codec_data::CodecData;
pub use compression::{Compressible, Encoder, ParCompressible, ParEncoder};
pub use decompression::{Decoder, Decompressible, ParDecoder, ParDecompressible};

// impl<I, T: DistanceValue, S: Cluster<T>, C: SquishyCluster<T, S>, M: Fn(&I, &I) -> T, Me, Enc: Encoder<I>, Dec: Decoder<I>>
//     Searchable<I, T, C, M> for CodecData<I, Me, Enc, Dec>
// {
//     fn query_to_center(&self, metric: &M, query: &I, cluster: &C) -> T {
//         metric(query, self.get(cluster.arg_center()))
//     }

//     fn query_to_all(&self, metric: &M, query: &I, cluster: &C) -> impl Iterator<Item = (usize, T)> {
//         let leaf_bytes = self.leaf_bytes();

//         cluster
//             .leaves()
//             .into_iter()
//             .map(C::arg_center)
//             .map(|o| {
//                 leaf_bytes
//                     .iter()
//                     .position(|(off, _)| *off == o)
//                     .unwrap_or_else(|| unreachable!("Offset not found in leaf offsets: {o}, {:?}", self.leaf_bytes()))
//             })
//             .map(|pos| &leaf_bytes[pos])
//             .flat_map(|(_, bytes)| self.decode_leaf(bytes, self.decoder()))
//             .map(|(i, p)| (i, metric(query, &p)))
//     }
// }

// impl<
//         I: Send + Sync,
//         T: DistanceValue + Send + Sync,
//         C: ParCluster<T>,
//         M: Fn(&I, &I) -> T + Send + Sync,
//         Me: Send + Sync,
//         Enc: ParEncoder<I>,
//         Dec: ParDecoder<I>,
//     > ParSearchable<I, T, SquishyBall<T, C>, M> for CodecData<I, Me, Enc, Dec>
// {
//     fn par_query_to_center(&self, metric: &M, query: &I, cluster: &SquishyBall<T, C>) -> T {
//         metric(query, self.get(cluster.arg_center()))
//     }

//     fn par_query_to_all(
//         &self,
//         metric: &M,
//         query: &I,
//         cluster: &SquishyBall<T, C>,
//     ) -> impl rayon::prelude::ParallelIterator<Item = (usize, T)> {
//         let leaf_bytes = self.leaf_bytes();

//         cluster
//             .leaves()
//             .into_par_iter()
//             .map(SquishyBall::arg_center)
//             .map(|o| {
//                 leaf_bytes
//                     .iter()
//                     .position(|(off, _)| *off == o)
//                     .unwrap_or_else(|| unreachable!("Offset not found in leaf offsets: {o}, {:?}", self.leaf_bytes()))
//             })
//             .map(|pos| &leaf_bytes[pos])
//             .flat_map(|(_, bytes)| self.decode_leaf(bytes, self.decoder()))
//             .map(|(i, p)| (i, metric(query, &p)))
//     }
// }
