//! An implementation of the Compression and Decompression traits.

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;

use crate::{
    cluster::ParCluster,
    dataset::{metric_space::ParMetricSpace, ParDataset},
    linear_search::{LinearSearch, ParLinearSearch},
    Cluster, Dataset, Metric, MetricSpace,
};

use super::{
    compression::ParCompressible, decompression::ParDecompressible, Compressible, Decodable, Decompressible, Encodable,
    SquishyBall,
};

/// A compressed dataset, that can be partially decompressed for search and
/// other applications.
///
/// A `CodecData` may only be built from a `Permutable` dataset, after the tree
/// has been built and the instances in the dataset have been permuted. This is
/// necessary for the `get` method to work correctly. Further, it is discouraged
/// to use the `get` method because it can be expensive if the instance being
/// retrieved is not the center of a cluster.
///
/// # Type Parameters
///
/// - `I`: The type of the instances in the dataset.
/// - `U`: The type of the numbers in the dataset.
/// - `M`: The type of the metadata associated with the instances.
pub struct CodecData<I, U, M> {
    /// The metric space of the dataset.
    pub(crate) metric: Metric<I, U>,
    /// The cardinality of the dataset.
    pub(crate) cardinality: usize,
    /// A hint for the dimensionality of the dataset.
    pub(crate) dimensionality_hint: (usize, Option<usize>),
    /// The metadata associated with the instances.
    pub(crate) metadata: Vec<M>,
    /// The centers of the clusters in the dataset.
    pub(crate) centers: HashMap<usize, I>,
    /// The bytes representing the leaf clusters as a flattened vector.
    pub(crate) leaf_bytes: Box<[u8]>,
    /// The offsets that indicate the start of the instances for each leaf
    /// cluster in the flattened vector.
    pub(crate) leaf_offsets: Vec<usize>,
    /// The cumulative cardinalities of the leaves.
    pub(crate) cumulative_cardinalities: Vec<usize>,
}

impl<I: Encodable + Decodable, U: Number> CodecData<I, U, usize> {
    /// Creates a `CodecData` from a compressible dataset and a `SquishyBall` tree.
    pub fn from_compressible<D: Compressible<I, U>, S: Cluster<I, U, D>>(
        data: &D,
        root: &SquishyBall<I, U, D, Self, S>,
    ) -> Self {
        let centers = root
            .subtree()
            .into_iter()
            .map(SquishyBall::arg_center)
            .map(|i| (i, data.get(i).clone()))
            .collect::<HashMap<_, _>>();

        let (leaf_bytes, leaf_offsets, cumulative_cardinalities) = data.encode_leaves(root);
        let cardinality = data.cardinality();
        let metric = data.metric().clone();
        let dimensionality_hint = data.dimensionality_hint();
        Self {
            metric,
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            centers,
            leaf_bytes,
            leaf_offsets,
            cumulative_cardinalities,
        }
    }
}

impl<I: Encodable + Decodable + Send + Sync, U: Number> CodecData<I, U, usize> {
    /// Creates a `CodecData` from a compressible dataset and a `SquishyBall` tree.
    pub fn par_from_compressible<D: ParCompressible<I, U>, S: ParCluster<I, U, D>>(
        data: &D,
        root: &SquishyBall<I, U, D, Self, S>,
    ) -> Self {
        let centers = root
            .subtree()
            .into_par_iter()
            .map(SquishyBall::arg_center)
            .map(|i| (i, data.get(i).clone()))
            .collect::<HashMap<_, _>>();

        let (leaf_bytes, leaf_offsets, cumulative_cardinalities) = data.par_encode_leaves(root);
        let cardinality = data.cardinality();
        let metric = data.metric().clone();
        let dimensionality_hint = data.dimensionality_hint();
        Self {
            metric,
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            centers,
            leaf_bytes,
            leaf_offsets,
            cumulative_cardinalities,
        }
    }
}

impl<I, U, M> CodecData<I, U, M> {
    /// Sets the metadata of the dataset.
    #[must_use]
    pub fn with_metadata<Mn>(self, metadata: Vec<Mn>) -> CodecData<I, U, Mn> {
        CodecData {
            metric: self.metric,
            cardinality: self.cardinality,
            dimensionality_hint: self.dimensionality_hint,
            metadata,
            centers: self.centers,
            leaf_bytes: self.leaf_bytes,
            leaf_offsets: self.leaf_offsets,
            cumulative_cardinalities: self.cumulative_cardinalities,
        }
    }

    /// Returns the metadata associated with the instances in the dataset.
    #[must_use]
    pub fn metadata(&self) -> &[M] {
        &self.metadata
    }
}

impl<I: Decodable, U: Number, M> Decompressible<I, U> for CodecData<I, U, M> {
    fn centers(&self) -> &HashMap<usize, I> {
        &self.centers
    }

    fn leaf_bytes(&self) -> &[u8] {
        self.leaf_bytes.as_ref()
    }

    fn leaf_offsets(&self) -> &[usize] {
        &self.leaf_offsets
    }

    /// Finds the offset of the leaf's instances in the compressed form, given
    /// the offset of the leaf in decompressed form.
    fn find_compressed_offset(&self, decompressed_offset: usize) -> usize {
        let pos = self
            .cumulative_cardinalities
            .iter()
            .position(|&i| i == decompressed_offset)
            .unwrap_or_else(|| unreachable!("Should be impossible to not hav the offset present here."));
        self.leaf_offsets[pos]
    }
}

impl<I: Decodable + Send + Sync, U: Number, M: Send + Sync> ParDecompressible<I, U> for CodecData<I, U, M> {}

impl<I, U: Number, M> Dataset<I, U> for CodecData<I, U, M> {
    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn dimensionality_hint(&self) -> (usize, Option<usize>) {
        self.dimensionality_hint
    }

    #[allow(clippy::panic)]
    fn get(&self, index: usize) -> &I {
        self.centers.get(&index).map_or_else(
            || panic!("For CodecData, the `get` method may only be used for cluster centers."),
            |center| center,
        )
    }
}

impl<I, U: Number, M> MetricSpace<I, U> for CodecData<I, U, M> {
    fn metric(&self) -> &Metric<I, U> {
        &self.metric
    }

    fn identity(&self) -> bool {
        self.metric.identity()
    }

    fn non_negativity(&self) -> bool {
        self.metric.non_negativity()
    }

    fn symmetry(&self) -> bool {
        self.metric.symmetry()
    }

    fn triangle_inequality(&self) -> bool {
        self.metric.triangle_inequality()
    }

    fn expensive(&self) -> bool {
        self.metric.expensive()
    }

    fn distance_function(&self) -> fn(&I, &I) -> U {
        self.metric.distance_function()
    }
}

impl<I: Decodable, U: Number, M> LinearSearch<I, U> for CodecData<I, U, M> {
    fn knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let mut knn = crate::linear_search::SizedHeap::new(Some(k));
        self.leaf_offsets
            .iter()
            .map(|&o| (o, self.decode_leaf(o)))
            .flat_map(|(o, instances)| {
                let instances = instances
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (o + i, p))
                    .collect::<Vec<_>>();
                MetricSpace::one_to_many(self, query, &instances)
            })
            .for_each(|(i, d)| knn.push((d, i)));
        knn.items().map(|(d, i)| (i, d)).collect()
    }

    fn rnn(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        self.leaf_offsets
            .iter()
            .map(|&o| (o, self.decode_leaf(o)))
            .flat_map(|(o, instances)| {
                let instances = instances
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (o + i, p))
                    .collect::<Vec<_>>();
                MetricSpace::one_to_many(self, query, &instances)
            })
            .filter(|&(_, d)| d <= radius)
            .collect()
    }
}

impl<I: Send + Sync, U: Number, M: Send + Sync> ParMetricSpace<I, U> for CodecData<I, U, M> {}

impl<I: Send + Sync, U: Number, M: Send + Sync> ParDataset<I, U> for CodecData<I, U, M> {}

impl<I: Decodable + Send + Sync, U: Number, M: Send + Sync> ParLinearSearch<I, U> for CodecData<I, U, M> {
    fn par_knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let mut knn = crate::linear_search::SizedHeap::new(Some(k));
        self.leaf_offsets
            .iter()
            .map(|&o| (o, self.decode_leaf(o)))
            .flat_map(|(o, instances)| {
                let instances = instances
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (o + i, p))
                    .collect::<Vec<_>>();
                ParMetricSpace::par_one_to_many(self, query, &instances)
            })
            .for_each(|(i, d)| knn.push((d, i)));
        knn.items().map(|(d, i)| (i, d)).collect()
    }

    fn par_rnn(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        self.leaf_offsets
            .iter()
            .map(|&o| (o, self.decode_leaf(o)))
            .flat_map(|(o, instances)| {
                let instances = instances
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (o + i, p))
                    .collect::<Vec<_>>();
                ParMetricSpace::par_one_to_many(self, query, &instances)
            })
            .filter(|&(_, d)| d <= radius)
            .collect()
    }
}
