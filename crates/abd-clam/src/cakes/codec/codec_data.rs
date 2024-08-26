//! An implementation of the Compression and Decompression traits.

use std::collections::HashMap;

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{
    cluster::ParCluster,
    dataset::{metric_space::ParMetricSpace, ParDataset, SizedHeap},
    Cluster, Dataset, FlatVec, Metric, MetricSpace, Permutable,
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
#[derive(Serialize, Deserialize)]
pub struct CodecData<I, U, M> {
    /// The metric space of the dataset.
    #[serde(skip)]
    pub(crate) metric: Metric<I, U>,
    /// The cardinality of the dataset.
    pub(crate) cardinality: usize,
    /// A hint for the dimensionality of the dataset.
    pub(crate) dimensionality_hint: (usize, Option<usize>),
    /// The metadata associated with the instances.
    #[serde(skip)]
    pub(crate) metadata: Vec<M>,
    /// The permutation of the original dataset.
    #[serde(skip)]
    pub(crate) permutation: Vec<usize>,
    /// The name of the dataset.
    pub(crate) name: String,
    /// The centers of the clusters in the dataset.
    pub(crate) center_map: HashMap<usize, I>,
    /// The byte-slices representing the leaf clusters.
    pub(crate) leaf_bytes: Vec<Box<[u8]>>,
    /// The offset-index of each leaf cluster.
    pub(crate) leaf_offsets: Vec<usize>,
}

impl<I: Encodable + Decodable, U: Number> CodecData<I, U, usize> {
    /// Creates a `CodecData` from a compressible dataset and a `SquishyBall` tree.
    pub fn from_compressible<Co: Compressible<I, U> + Permutable, S: Cluster<I, U, Co>>(
        data: &Co,
        root: &SquishyBall<I, U, Co, Self, S>,
    ) -> Self {
        let center_map = root
            .subtree()
            .into_iter()
            .map(Cluster::arg_center)
            .map(|i| (i, data.get(i).clone()))
            .collect();

        let leaf_bytes = data.encode_leaves(root);
        let leaf_offsets = root.leaves().iter().map(|leaf| leaf.offset()).collect();

        let cardinality = data.cardinality();
        let metric = data.metric().clone();
        let dimensionality_hint = data.dimensionality_hint();
        Self {
            metric,
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            permutation: data.permutation(),
            name: format!("CodecData({})", data.name()),
            center_map,
            leaf_bytes,
            leaf_offsets,
        }
    }
}

impl<I: Encodable + Decodable + Send + Sync, U: Number> CodecData<I, U, usize> {
    /// Creates a `CodecData` from a compressible dataset and a `SquishyBall` tree.
    pub fn par_from_compressible<D: ParCompressible<I, U> + Permutable, S: ParCluster<I, U, D> + core::fmt::Debug>(
        data: &D,
        root: &SquishyBall<I, U, D, Self, S>,
    ) -> Self {
        let center_map = root
            .subtree()
            .into_iter()
            .map(Cluster::arg_center)
            .map(|i| (i, data.get(i).clone()))
            .collect();

        let leaf_bytes = data.par_encode_leaves(root);
        let leaf_offsets = root.leaves().iter().map(|leaf| leaf.offset()).collect();

        let cardinality = data.cardinality();
        let metric = data.metric().clone();
        let dimensionality_hint = data.dimensionality_hint();
        Self {
            metric,
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            permutation: data.permutation(),
            name: format!("CodecData({})", data.name()),
            center_map,
            leaf_bytes,
            leaf_offsets,
        }
    }
}

impl<I, U, M> CodecData<I, U, M> {
    /// Sets the permutation and metadata of the dataset after deserialization.
    ///
    /// # Parameters
    ///
    /// - `permutation`: The permutation of the original dataset.
    /// - `metadata`: The new metadata to associate with the instances.
    ///
    /// # Type Parameters
    ///
    /// - `Me`: The type of the new metadata.
    ///
    /// # Returns
    ///
    /// A `CodecData` with the permutation and metadata set.
    ///
    /// # Errors
    ///
    /// - If the length of the permutation vector does not match the cardinality
    ///   of the dataset.
    /// - If the length of the metadata vector does not match the cardinality of
    ///   the dataset.
    pub fn post_deserialization<Me>(
        mut self,
        permutation: Vec<usize>,
        metadata: Vec<Me>,
    ) -> Result<CodecData<I, U, Me>, String> {
        if permutation.len() == self.cardinality {
            self.permutation = permutation;
            self.with_metadata(metadata)
        } else {
            Err(format!(
                "The length of the permutation vector ({}) does not match the cardinality of the dataset ({}).",
                permutation.len(),
                self.cardinality
            ))
        }
    }

    /// Changes the metadata of the dataset.
    ///
    /// # Parameters
    ///
    /// - `metadata`: The new metadata to associate with the instances.
    ///
    /// # Type Parameters
    ///
    /// - `Me`: The type of the new metadata.
    ///
    /// # Returns
    ///
    /// A `CodecData` with the new metadata.
    ///
    /// # Errors
    ///
    /// If the length of the metadata vector does not match the cardinality of
    /// the dataset.
    pub fn with_metadata<Me>(self, mut metadata: Vec<Me>) -> Result<CodecData<I, U, Me>, String> {
        if metadata.len() == self.cardinality {
            metadata.permute(&self.permutation);
            Ok(CodecData {
                metric: self.metric,
                cardinality: self.cardinality,
                dimensionality_hint: self.dimensionality_hint,
                metadata,
                permutation: self.permutation,
                name: self.name,
                center_map: self.center_map,
                leaf_bytes: self.leaf_bytes,
                leaf_offsets: self.leaf_offsets,
            })
        } else {
            Err(format!(
                "The length of the metadata vector ({}) does not match the cardinality of the dataset ({}).",
                metadata.len(),
                self.cardinality
            ))
        }
    }

    /// Returns the metadata associated with the instances in the dataset.
    #[must_use]
    pub fn metadata(&self) -> &[M] {
        &self.metadata
    }

    /// Returns the permutation of the original dataset.
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
}

impl<I: Decodable + Clone, U: Number, M: Clone> CodecData<I, U, M> {
    /// Decompresses the dataset into a `FlatVec`.
    #[must_use]
    pub fn to_flat_vec(&self) -> FlatVec<I, U, M> {
        let instances = self
            .leaf_bytes
            .iter()
            .flat_map(|bytes| self.decode_leaf(bytes.as_ref()))
            .collect::<Vec<_>>();
        FlatVec {
            metric: self.metric.clone(),
            instances,
            dimensionality_hint: self.dimensionality_hint,
            permutation: self.permutation.clone(),
            metadata: self.metadata.clone(),
            name: format!("FlatVec({})", self.name),
        }
    }
}

impl<I: Decodable, U: Number, M> Decompressible<I, U> for CodecData<I, U, M> {
    fn centers(&self) -> &HashMap<usize, I> {
        &self.center_map
    }

    fn leaf_bytes(&self) -> &[Box<[u8]>] {
        self.leaf_bytes.as_ref()
    }

    fn leaf_offsets(&self) -> &[usize] {
        &self.leaf_offsets
    }
}

impl<I: Decodable + Send + Sync, U: Number, M: Send + Sync> ParDecompressible<I, U> for CodecData<I, U, M> {}

impl<I: Decodable, U: Number, M> Dataset<I, U> for CodecData<I, U, M> {
    fn name(&self) -> &str {
        &self.name
    }

    fn with_name(mut self, name: &str) -> Self {
        self.name = format!("CodecData({name})");
        self
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn dimensionality_hint(&self) -> (usize, Option<usize>) {
        self.dimensionality_hint
    }

    #[allow(clippy::panic)]
    fn get(&self, index: usize) -> &I {
        self.center_map.get(&index).map_or_else(
            || panic!("For CodecData, the `get` method may only be used for cluster centers."),
            |center| center,
        )
    }

    fn knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let mut knn = SizedHeap::new(Some(k));
        self.leaf_bytes
            .iter()
            .map(|bytes| self.decode_leaf(bytes.as_ref()))
            .zip(self.leaf_offsets.iter())
            .flat_map(|(instances, &o)| {
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
        self.leaf_bytes
            .iter()
            .map(|bytes| self.decode_leaf(bytes.as_ref()))
            .zip(self.leaf_offsets.iter())
            .flat_map(|(instances, &o)| {
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

impl<I, U: Number, M> MetricSpace<I, U> for CodecData<I, U, M> {
    fn metric(&self) -> &Metric<I, U> {
        &self.metric
    }

    fn set_metric(&mut self, metric: Metric<I, U>) {
        self.metric = metric;
    }
}

impl<I: Send + Sync, U: Number, M: Send + Sync> ParMetricSpace<I, U> for CodecData<I, U, M> {}

impl<I: Decodable + Send + Sync, U: Number, M: Send + Sync> ParDataset<I, U> for CodecData<I, U, M> {
    fn par_knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let mut knn = SizedHeap::new(Some(k));
        self.leaf_bytes
            .iter()
            .map(|bytes| self.decode_leaf(bytes.as_ref()))
            .zip(self.leaf_offsets.iter())
            .flat_map(|(instances, &o)| {
                let instances = instances
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (o + i, p))
                    .collect::<Vec<_>>();
                ParMetricSpace::par_one_to_many(self, query, &instances)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(i, d)| knn.push((d, i)));
        knn.items().map(|(d, i)| (i, d)).collect()
    }

    fn par_rnn(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        self.leaf_bytes
            .iter()
            .map(|bytes| self.decode_leaf(bytes.as_ref()))
            .zip(self.leaf_offsets.iter())
            .flat_map(|(instances, &o)| {
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
