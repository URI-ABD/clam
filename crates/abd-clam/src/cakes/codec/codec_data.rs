//! An implementation of the Compression and Decompression traits.

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;
use serde::{
    de::Deserializer,
    ser::{SerializeStruct, Serializer},
    Deserialize, Serialize,
};

use crate::{
    cluster::ParCluster,
    dataset::{metric_space::ParMetricSpace, ParDataset},
    linear_search::{LinearSearch, ParLinearSearch},
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
pub struct CodecData<I, U, M> {
    /// The metric space of the dataset.
    pub(crate) metric: Metric<I, U>,
    /// The cardinality of the dataset.
    pub(crate) cardinality: usize,
    /// A hint for the dimensionality of the dataset.
    pub(crate) dimensionality_hint: (usize, Option<usize>),
    /// The metadata associated with the instances.
    pub(crate) metadata: Vec<M>,
    /// The permutation of the original dataset.
    pub(crate) permutation: Vec<usize>,
    /// The centers of the clusters in the dataset.
    pub(crate) center_map: HashMap<usize, I>,
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

        let (leaf_bytes, leaf_offsets, cumulative_cardinalities) = data.encode_leaves(root);
        let cardinality = data.cardinality();
        let metric = data.metric().clone();
        let dimensionality_hint = data.dimensionality_hint();
        Self {
            metric,
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            permutation: data.permutation(),
            center_map,
            leaf_bytes,
            leaf_offsets,
            cumulative_cardinalities,
        }
    }
}

impl<I: Encodable + Decodable + Send + Sync, U: Number> CodecData<I, U, usize> {
    /// Creates a `CodecData` from a compressible dataset and a `SquishyBall` tree.
    pub fn par_from_compressible<D: ParCompressible<I, U> + Permutable, S: ParCluster<I, U, D>>(
        data: &D,
        root: &SquishyBall<I, U, D, Self, S>,
    ) -> Self {
        let center_map = root
            .subtree()
            .into_iter()
            .map(Cluster::arg_center)
            .map(|i| (i, data.get(i).clone()))
            .collect();

        let (leaf_bytes, leaf_offsets, cumulative_cardinalities) = data.par_encode_leaves(root);
        let cardinality = data.cardinality();
        let metric = data.metric().clone();
        let dimensionality_hint = data.dimensionality_hint();
        Self {
            metric,
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            permutation: data.permutation(),
            center_map,
            leaf_bytes,
            leaf_offsets,
            cumulative_cardinalities,
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
                center_map: self.center_map,
                leaf_bytes: self.leaf_bytes,
                leaf_offsets: self.leaf_offsets,
                cumulative_cardinalities: self.cumulative_cardinalities,
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
            .leaf_offsets
            .iter()
            .flat_map(|&o| self.decode_leaf(o))
            .collect::<Vec<_>>();
        FlatVec {
            metric: self.metric.clone(),
            instances,
            dimensionality_hint: self.dimensionality_hint,
            permutation: self.permutation.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

impl<I: Decodable, U: Number, M> Decompressible<I, U> for CodecData<I, U, M> {
    fn centers(&self) -> &HashMap<usize, I> {
        &self.center_map
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
        self.center_map.get(&index).map_or_else(
            || panic!("For CodecData, the `get` method may only be used for cluster centers."),
            |center| center,
        )
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

impl<I: Decodable, U: Number, M> LinearSearch<I, U> for CodecData<I, U, M> {
    fn knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let mut knn = crate::linear_search::SizedHeap::new(Some(k));
        self.leaf_offsets
            .iter()
            .map(|&o| self.decode_leaf(o))
            .zip(self.cumulative_cardinalities.iter())
            .flat_map(|(instances, o)| {
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
            .map(|&o| self.decode_leaf(o))
            .zip(self.cumulative_cardinalities.iter())
            .flat_map(|(instances, o)| {
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
            .par_iter()
            .map(|&o| self.decode_leaf(o))
            .zip(self.cumulative_cardinalities.par_iter())
            .flat_map(|(instances, o)| {
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
        self.leaf_offsets
            .par_iter()
            .map(|&o| self.decode_leaf(o))
            .zip(self.cumulative_cardinalities.par_iter())
            .flat_map(|(instances, o)| {
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

/// Returns binary encodings for the centers of the clusters in the dataset.
fn encode_centers<I: Encodable>(center_map: &HashMap<usize, I>) -> Box<[u8]> {
    // TODO: Bring back recursive encodings after fixing graph traversal for
    // when multiple clusters have the same arg_center.
    let mut bytes = Vec::new();

    for (i, center) in center_map {
        bytes.extend(i.to_le_bytes());
        let encoding = center.as_bytes();
        bytes.extend(encoding.len().to_le_bytes());
        bytes.extend(encoding);
    }

    bytes.into_boxed_slice()
}

/// Decodes the binary encodings for the centers of the clusters in the dataset.
fn decode_centers<I: Decodable>(bytes: &[u8]) -> HashMap<usize, I> {
    let mut offset = 0;

    let mut center_map = HashMap::new();
    while offset < bytes.len() {
        let i = super::read_usize(bytes, &mut offset);
        let encoding = super::read_encoding(bytes, &mut offset);
        let center = I::from_bytes(&encoding);
        center_map.insert(i, center);
    }

    center_map
}

/// This struct is used for serializing and deserializing a `CodecData`.
#[derive(Serialize, Deserialize)]
struct CodecDataSerde {
    /// The cardinality of the dataset.
    cardinality: usize,
    /// A hint for the dimensionality of the dataset.
    dimensionality_hint: (usize, Option<usize>),
    /// The bytes representing the centers of the clusters in the dataset.
    center_bytes: Box<[u8]>,
    /// The bytes representing the leaf clusters as a flattened vector.
    leaf_bytes: Box<[u8]>,
    /// The offsets that indicate the start of the instances for each leaf
    leaf_offsets: Vec<usize>,
    /// The cumulative cardinalities of the leaves.
    cumulative_cardinalities: Vec<usize>,
}

impl<I: Encodable, U, M> Serialize for CodecData<I, U, M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let center_bytes = encode_centers(&self.center_map);
        let mut state = serializer.serialize_struct("CodecDataSerde", 6)?;
        state.serialize_field("cardinality", &self.cardinality)?;
        state.serialize_field("dimensionality_hint", &self.dimensionality_hint)?;
        state.serialize_field("center_bytes", &center_bytes)?;
        state.serialize_field("leaf_bytes", &self.leaf_bytes)?;
        state.serialize_field("leaf_offsets", &self.leaf_offsets)?;
        state.serialize_field("cumulative_cardinalities", &self.cumulative_cardinalities)?;
        state.end()
    }
}

impl<'de, I: Decodable, U> Deserialize<'de> for CodecData<I, U, usize> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let CodecDataSerde {
            cardinality,
            dimensionality_hint,
            center_bytes,
            leaf_bytes,
            leaf_offsets,
            cumulative_cardinalities,
        } = CodecDataSerde::deserialize(deserializer)?;
        let center_map = decode_centers(&center_bytes);
        let permutation = (0..cardinality).collect::<Vec<_>>();
        Ok(Self {
            metric: Metric::default(),
            cardinality,
            dimensionality_hint,
            metadata: permutation.clone(),
            permutation,
            center_map,
            leaf_bytes,
            leaf_offsets,
            cumulative_cardinalities,
        })
    }
}
