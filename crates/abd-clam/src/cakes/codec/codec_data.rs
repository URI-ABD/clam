//! An implementation of the Compression and Decompression traits.

use core::marker::PhantomData;
use std::{
    collections::HashMap,
    io::{Read, Write},
};

use distances::Number;
use flate2::write::GzEncoder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    cakes::codec::{read_encoding, read_usize},
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
#[derive(Clone)]
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
    /// The name of the dataset.
    pub(crate) name: String,
    /// The centers of the clusters in the dataset.
    pub(crate) center_map: HashMap<usize, I>,
    /// The byte-slices representing the leaf clusters.
    pub(crate) leaf_bytes: Vec<(usize, Box<[u8]>)>,
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

        let leaf_bytes = data
            .encode_leaves(root)
            .into_iter()
            .map(|(leaf, bytes)| (leaf.offset(), bytes))
            .collect();

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

        let leaf_bytes = data
            .par_encode_leaves(root)
            .into_iter()
            .map(|(leaf, bytes)| (leaf.offset(), bytes))
            .collect();

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
        }
    }
}

impl<I, U, M> CodecData<I, U, M> {
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
            .flat_map(|(_, bytes)| self.decode_leaf(bytes.as_ref()))
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

    fn leaf_bytes(&self) -> &[(usize, Box<[u8]>)] {
        &self.leaf_bytes
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
            .map(|(o, bytes)| (*o, self.decode_leaf(bytes.as_ref())))
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
        self.leaf_bytes
            .iter()
            .map(|(o, bytes)| (*o, self.decode_leaf(bytes.as_ref())))
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
            .par_iter()
            .map(|(o, bytes)| (*o, self.decode_leaf(bytes.as_ref())))
            .flat_map(|(o, instances)| {
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
            .par_iter()
            .map(|(o, bytes)| (*o, self.decode_leaf(bytes.as_ref())))
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

/// A private helper struct for serializing and deserializing `CodecData`.
#[derive(Serialize, Deserialize)]
struct CodecDataSerde<I, U, M> {
    /// The cardinality of the dataset.
    cardinality: usize,
    /// A hint for the dimensionality of the dataset.
    dimensionality_hint: (usize, Option<usize>),
    /// The name of the dataset.
    name: String,
    /// The number of bytes for the `metadata`.
    num_metadata: usize,
    /// The number of bytes for the `center_map`.
    num_center_map: usize,
    /// The number of bytes for the `leaf_bytes`.
    num_leaf_bytes: usize,
    /// The compressed bytes.
    bytes: Box<[u8]>,
    /// Phantom data.
    _p: PhantomData<(I, U, M)>,
}

impl<I: Encodable + Send + Sync, U, M: Encodable + Send + Sync> CodecDataSerde<I, U, M> {
    /// Creates a `CodecDataSerde` from a `CodecData`.
    fn from_codec_data(data: &CodecData<I, U, M>) -> Result<Self, String> {
        let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());

        let metadata = data
            .metadata
            .par_iter()
            .flat_map(|m| {
                let mut bytes = Vec::new();
                let encoding = m.as_bytes();
                bytes.extend_from_slice(&encoding.len().to_le_bytes());
                bytes.extend_from_slice(&encoding);
                bytes
            })
            .collect::<Vec<_>>();
        let num_metadata = metadata.len();
        encoder.write_all(&metadata).map_err(|e| e.to_string())?;

        let permutation = data
            .permutation
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<_>>();
        encoder.write_all(&permutation).map_err(|e| e.to_string())?;

        let center_map = data
            .center_map
            .par_iter()
            .flat_map(|(i, p)| {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&i.to_le_bytes());
                let encoding = p.as_bytes();
                bytes.extend_from_slice(&encoding.len().to_le_bytes());
                bytes.extend_from_slice(&encoding);
                bytes
            })
            .collect::<Vec<_>>();
        let num_center_map = center_map.len();
        encoder.write_all(&center_map).map_err(|e| e.to_string())?;

        let leaf_bytes = data
            .leaf_bytes
            .par_iter()
            .flat_map(|(i, encodings)| {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&i.to_le_bytes());
                bytes.extend_from_slice(&encodings.len().to_le_bytes());
                bytes.extend_from_slice(encodings);
                bytes
            })
            .collect::<Vec<_>>();
        let num_leaf_bytes = leaf_bytes.len();
        encoder.write_all(&leaf_bytes).map_err(|e| e.to_string())?;

        let bytes = encoder.finish().map_err(|e| e.to_string())?.into_boxed_slice();

        Ok(Self {
            cardinality: data.cardinality,
            dimensionality_hint: data.dimensionality_hint,
            name: data.name.clone(),
            num_metadata,
            num_center_map,
            num_leaf_bytes,
            bytes,
            _p: PhantomData,
        })
    }
}

impl<I: Decodable + Send + Sync, U, M: Decodable + Send + Sync> CodecData<I, U, M> {
    /// Creates a `CodecData` from a `CodecDataSerde`.
    fn from_serde(data: CodecDataSerde<I, U, M>) -> Result<Self, String> {
        let mut decoder = flate2::read::GzDecoder::new(&data.bytes[..]);

        let (metadata, permutation, center_map, leaf_bytes) = {
            let metadata = {
                let mut bytes = vec![0_u8; data.num_metadata];
                decoder.read_exact(&mut bytes).map_err(|e| e.to_string())?;
                bytes
            };

            let permutation = {
                let mut bytes = vec![0_u8; data.cardinality * core::mem::size_of::<usize>()];
                decoder.read_exact(&mut bytes).map_err(|e| e.to_string())?;
                bytes
            };

            let center_map = {
                let mut bytes = vec![0_u8; data.num_center_map];
                decoder.read_exact(&mut bytes).map_err(|e| e.to_string())?;
                bytes
            };

            let leaf_bytes = {
                let mut bytes = vec![0_u8; data.num_leaf_bytes];
                decoder.read_exact(&mut bytes).map_err(|e| e.to_string())?;
                bytes
            };

            (metadata, permutation, center_map, leaf_bytes)
        };

        let (metadata, (permutation, (center_map, leaf_bytes))) = rayon::join(
            || Self::decode_metadata(&metadata),
            || {
                rayon::join(
                    || Self::decode_permutation(&permutation),
                    || {
                        rayon::join(
                            || Self::decode_center_map(&center_map),
                            || Self::decode_leaf_bytes(&leaf_bytes),
                        )
                    },
                )
            },
        );

        Ok(Self {
            metric: Metric::default(),
            cardinality: data.cardinality,
            dimensionality_hint: data.dimensionality_hint,
            metadata,
            permutation,
            name: data.name,
            center_map,
            leaf_bytes,
        })
    }

    /// Decodes the metadata from the compressed bytes.
    fn decode_metadata(bytes: &[u8]) -> Vec<M> {
        let mut metadata = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let encoding = read_encoding(bytes, &mut offset);
            metadata.push(M::from_bytes(&encoding));
        }
        metadata
    }

    /// Decodes the permutation from the compressed bytes.
    fn decode_permutation(bytes: &[u8]) -> Vec<usize> {
        bytes
            .chunks_exact(core::mem::size_of::<usize>())
            .map(|chunk| {
                let mut array = [0; std::mem::size_of::<usize>()];
                array.copy_from_slice(&chunk[..std::mem::size_of::<usize>()]);
                usize::from_le_bytes(array)
            })
            .collect::<Vec<_>>()
    }

    /// Decodes the center map from the compressed bytes.
    fn decode_center_map(bytes: &[u8]) -> HashMap<usize, I> {
        let mut center_map = HashMap::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let i = read_usize(bytes, &mut offset);
            let encoding = read_encoding(bytes, &mut offset);
            center_map.insert(i, I::from_bytes(&encoding));
        }
        center_map
    }

    /// Decodes the leaf bytes from the compressed bytes.
    fn decode_leaf_bytes(bytes: &[u8]) -> Vec<(usize, Box<[u8]>)> {
        let mut leaf_bytes = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let i = read_usize(bytes, &mut offset);
            let encoding = read_encoding(bytes, &mut offset);
            leaf_bytes.push((i, encoding));
        }
        leaf_bytes
    }
}

impl<I: Encodable + Send + Sync, U, M: Encodable + Send + Sync> Serialize for CodecData<I, U, M> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CodecDataSerde::from_codec_data(self)
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, I: Decodable + Send + Sync, U, M: Decodable + Send + Sync> Deserialize<'de> for CodecData<I, U, M> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        CodecDataSerde::deserialize(deserializer)
            .and_then(|serde| Self::from_serde(serde).map_err(serde::de::Error::custom))
    }
}
