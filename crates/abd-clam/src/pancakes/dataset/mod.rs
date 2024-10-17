//! Datasets for `PanCAKES`.

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;

use crate::{
    cakes::{HintedDataset, ParHintedDataset, ParSearchable, Searchable},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Dataset, FlatVec, Metric,
};

use super::SquishyBall;

mod codec_data;
mod compression;
mod decompression;

pub use codec_data::CodecData;
pub use compression::{Compressible, Encodable, ParCompressible};
pub use decompression::{Decodable, Decompressible, ParDecompressible};

impl<I: Encodable, Me> Compressible<I> for FlatVec<I, Me> {}
impl<I: Encodable + Send + Sync, Me: Send + Sync> ParCompressible<I> for FlatVec<I, Me> {}

impl<I: Decodable, T: Number, C: Cluster<T>, M: Metric<I, T>, Me> Searchable<I, T, SquishyBall<T, C>, M>
    for CodecData<I, Me>
{
    fn query_to_center(&self, metric: &M, query: &I, cluster: &SquishyBall<T, C>) -> T {
        metric.distance(query, self.get(cluster.arg_center()))
    }

    fn query_to_all(&self, metric: &M, query: &I, cluster: &SquishyBall<T, C>) -> impl Iterator<Item = (usize, T)> {
        let leaf_bytes = self.leaf_bytes();

        cluster
            .leaves()
            .into_iter()
            .map(SquishyBall::offset)
            .map(|o| {
                leaf_bytes
                    .iter()
                    .position(|(off, _)| *off == o)
                    .unwrap_or_else(|| unreachable!("Offset not found in leaf offsets: {o}, {:?}", self.leaf_bytes()))
            })
            .map(|pos| &leaf_bytes[pos])
            .flat_map(|(o, bytes)| {
                self.decode_leaf(bytes)
                    .into_iter()
                    .enumerate()
                    .map(|(i, p)| (i + *o, p))
            })
            .map(|(i, p)| (i, metric.distance(query, &p)))
    }
}

impl<I: Decodable + Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, Me: Send + Sync>
    ParSearchable<I, T, SquishyBall<T, C>, M> for CodecData<I, Me>
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
            .map(SquishyBall::offset)
            .map(|o| {
                leaf_bytes
                    .iter()
                    .position(|(off, _)| *off == o)
                    .unwrap_or_else(|| unreachable!("Offset not found in leaf offsets: {o}, {:?}", self.leaf_bytes()))
            })
            .map(|pos| &leaf_bytes[pos])
            .flat_map(|(o, bytes)| {
                self.decode_leaf(bytes)
                    .into_par_iter()
                    .enumerate()
                    .map(|(i, p)| (i + *o, p))
            })
            .map(|(i, p)| (i, metric.par_distance(query, &p)))
    }
}

#[allow(clippy::implicit_hasher)]
impl<I: Decodable, T: Number, C: Cluster<T>, M: Metric<I, T>, Me> HintedDataset<I, T, SquishyBall<T, C>, M>
    for CodecData<I, (Me, HashMap<usize, T>)>
{
    fn hints_for(&self, i: usize) -> &HashMap<usize, T> {
        &self.metadata[i].1
    }

    fn hints_for_mut(&mut self, i: usize) -> &mut HashMap<usize, T> {
        &mut self.metadata[i].1
    }
}

#[allow(clippy::implicit_hasher)]
impl<I: Decodable + Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, Me: Send + Sync>
    ParHintedDataset<I, T, SquishyBall<T, C>, M> for CodecData<I, (Me, HashMap<usize, T>)>
{
}
