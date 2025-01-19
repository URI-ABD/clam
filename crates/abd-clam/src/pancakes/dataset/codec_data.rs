//! An implementation of the `Compression` and `Decompression` traits on a
//! `Dataset`.

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;

use crate::{
    cluster::ParCluster,
    dataset::{AssociatesMetadata, AssociatesMetadataMut, ParDataset, Permutable},
    Cluster, Dataset, FlatVec,
};

use super::{
    super::SquishyBall,
    compression::{Compressible, Encodable, ParCompressible},
    decompression::{Decodable, Decompressible, ParDecompressible},
};

#[cfg(feature = "disk-io")]
use std::io::{Read, Write};

#[cfg(feature = "disk-io")]
use flate2::{read::GzDecoder, write::GzEncoder, Compression};

/// A compressed dataset, that can be partially decompressed for search and
/// other applications.
///
/// # Type Parameters
///
/// - `I`: The type of the items in the dataset.
/// - `Me`: The type of the metadata associated with the items.
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
pub struct CodecData<I, Me> {
    /// The cardinality of the dataset.
    cardinality: usize,
    /// A hint for the dimensionality of the dataset.
    dimensionality_hint: (usize, Option<usize>),
    /// The metadata associated with the items.
    pub(crate) metadata: Vec<Me>,
    /// The permutation of the original dataset.
    permutation: Vec<usize>,
    /// The name of the dataset.
    name: String,
    /// The centers of the clusters in the dataset.
    center_map: HashMap<usize, I>,
    /// The byte-slices representing the leaf clusters.
    leaf_bytes: Vec<(usize, Box<[u8]>)>,
}

impl<I: Encodable + Clone> CodecData<I, usize> {
    /// Creates a `CodecData` from a compressible dataset and a `SquishyBall` tree.
    pub fn from_compressible<T: Number, D: Compressible<I>, S: Cluster<T>>(data: &D, root: &SquishyBall<T, S>) -> Self {
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
        let dimensionality_hint = data.dimensionality_hint();

        Self {
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            permutation: (0..cardinality).collect(),
            name: format!("CodecData({})", data.name()),
            center_map,
            leaf_bytes,
        }
    }
}

impl<I: Encodable + Clone + Send + Sync> CodecData<I, usize> {
    /// Parallel version of [`CodecData::from_compressible`](crate::pancakes::dataset::CodecData::from_compressible).
    pub fn par_from_compressible<T: Number, D: ParCompressible<I>, S: ParCluster<T>>(
        data: &D,
        root: &SquishyBall<T, S>,
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
        let dimensionality_hint = data.dimensionality_hint();

        Self {
            cardinality,
            dimensionality_hint,
            metadata: (0..cardinality).collect(),
            permutation: (0..cardinality).collect(),
            name: format!("CodecData({})", data.name()),
            center_map,
            leaf_bytes,
        }
    }
}

impl<I, Me> CodecData<I, Me> {
    /// Changes the permutation of the dataset without changing the order of the
    /// items.
    #[must_use]
    pub fn with_permutation(self, permutation: &[usize]) -> Self {
        Self {
            cardinality: self.cardinality,
            dimensionality_hint: self.dimensionality_hint,
            metadata: self.metadata,
            permutation: permutation.to_vec(),
            name: self.name,
            center_map: self.center_map,
            leaf_bytes: self.leaf_bytes,
        }
    }

    /// Returns the permutation of the original dataset.
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Returns the center map of the dataset.
    ///
    /// This is a map from the index of the center (in the decompressed version)
    /// to the center itself.
    #[must_use]
    pub const fn center_map(&self) -> &HashMap<usize, I> {
        &self.center_map
    }

    /// Returns the leaf bytes of the dataset.
    ///
    /// This is an array of tuples of the offset of the leaf cluster and the
    /// compressed bytes of the items in the leaf cluster.
    #[must_use]
    pub fn leaf_bytes(&self) -> &[(usize, Box<[u8]>)] {
        &self.leaf_bytes
    }

    /// Transforms the centers of the dataset using the given function.
    pub fn transform_centers<It, F: Fn(I) -> It>(self, transformer: F) -> CodecData<It, Me> {
        let center_map = self
            .center_map
            .into_iter()
            .map(|(i, center)| (i, transformer(center)))
            .collect();

        CodecData {
            cardinality: self.cardinality,
            dimensionality_hint: self.dimensionality_hint,
            metadata: self.metadata,
            permutation: self.permutation,
            name: self.name,
            center_map,
            leaf_bytes: self.leaf_bytes,
        }
    }
}

impl<I: Decodable + Clone, Me: Clone> CodecData<I, Me> {
    /// Decompresses the dataset into a `FlatVec`.
    ///
    /// # Errors
    ///
    /// - If the `FlatVec` cannot be created from the decompressed items.
    pub fn to_flat_vec(&self) -> Result<FlatVec<I, Me>, String> {
        let items = self
            .leaf_bytes
            .iter()
            .flat_map(|(_, bytes)| self.decode_leaf(bytes.as_ref()))
            .collect::<Vec<_>>();

        let (min_dim, max_dim) = self.dimensionality_hint;

        let data = FlatVec::new(items)?
            .with_name(&self.name)
            .with_metadata(&self.metadata)?
            .with_permutation(&self.permutation)
            .with_dim_lower_bound(min_dim);

        let data = if let Some(max_dim) = max_dim {
            data.with_dim_upper_bound(max_dim)
        } else {
            data
        };

        Ok(data)
    }
}

impl<I: Decodable + Clone + Send + Sync, Me: Clone + Send + Sync> CodecData<I, Me> {
    /// Parallel version of [`CodecData::to_flat_vec`](crate::pancakes::dataset::CodecData::to_flat_vec).
    ///
    /// # Errors
    ///
    /// See [`CodecData::to_flat_vec`](crate::pancakes::dataset::CodecData::to_flat_vec).
    pub fn par_to_flat_vec(&self) -> Result<FlatVec<I, Me>, String> {
        let items = self
            .leaf_bytes
            .par_iter()
            .flat_map(|(_, bytes)| self.decode_leaf(bytes.as_ref()))
            .collect::<Vec<_>>();

        let (min_dim, max_dim) = self.dimensionality_hint;

        let data = FlatVec::new(items)?
            .with_name(&self.name)
            .with_metadata(&self.metadata)?
            .with_permutation(&self.permutation)
            .with_dim_lower_bound(min_dim);

        let data = if let Some(max_dim) = max_dim {
            data.with_dim_upper_bound(max_dim)
        } else {
            data
        };

        Ok(data)
    }
}

impl<I: Decodable, Me> Dataset<I> for CodecData<I, Me> {
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
}

impl<I: Decodable + Send + Sync, Me: Send + Sync> ParDataset<I> for CodecData<I, Me> {}

impl<I: Decodable, Me> AssociatesMetadata<I, Me> for CodecData<I, Me> {
    fn metadata(&self) -> &[Me] {
        &self.metadata
    }

    fn metadata_at(&self, index: usize) -> &Me {
        &self.metadata[index]
    }
}

impl<I: Decodable, Me, Met: Clone> AssociatesMetadataMut<I, Me, Met, CodecData<I, Met>> for CodecData<I, Me> {
    fn metadata_mut(&mut self) -> &mut [Me] {
        &mut self.metadata
    }

    fn metadata_at_mut(&mut self, index: usize) -> &mut Me {
        &mut self.metadata[index]
    }

    fn with_metadata(self, metadata: &[Met]) -> Result<CodecData<I, Met>, String> {
        if metadata.len() == self.cardinality {
            let mut metadata = metadata.to_vec();
            metadata.permute(&self.permutation);
            Ok(CodecData {
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

    fn transform_metadata<F: Fn(&Me) -> Met>(self, f: F) -> CodecData<I, Met> {
        let metadata = self.metadata.iter().map(f).collect();
        CodecData {
            cardinality: self.cardinality,
            dimensionality_hint: self.dimensionality_hint,
            metadata,
            permutation: self.permutation,
            name: self.name,
            center_map: self.center_map,
            leaf_bytes: self.leaf_bytes,
        }
    }
}

#[cfg(feature = "disk-io")]
/// Encodes using bitcode and compresses using Gzip.
///
/// # Errors
///
/// - If the item cannot be encoded.
/// - If the encoded bytes cannot be compressed.
fn encode_and_compress<T: bitcode::Encode>(item: T) -> Result<Vec<u8>, String> {
    let buf = bitcode::encode(&item).map_err(|e| e.to_string())?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&buf).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

#[cfg(feature = "disk-io")]
/// Decompresses using Gzip and decodes using bitcode.
///
/// # Errors
///
/// - If the bytes cannot be decompressed.
/// - If the decompressed bytes cannot be decoded.
fn decompress_and_decode<T: bitcode::Decode>(bytes: &[u8]) -> Result<T, String> {
    let mut decoder = GzDecoder::new(bytes);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    bitcode::decode(&buf).map_err(|e| e.to_string())
}

#[cfg(feature = "disk-io")]
impl<I: Decodable + bitcode::Encode + bitcode::Decode, Me: bitcode::Encode + bitcode::Decode>
    crate::dataset::DatasetIO<I> for CodecData<I, Me>
{
    fn write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let metadata_bytes = encode_and_compress(&self.metadata)?;

        let permutation_bytes = encode_and_compress(&self.permutation)?;

        let center_map = self
            .center_map
            .iter()
            .map(|(&i, p)| (i, p.as_bytes()))
            .collect::<Vec<_>>();
        let center_map_bytes = encode_and_compress(center_map)?;

        let leaf_bytes = encode_and_compress(&self.leaf_bytes)?;

        let members = (
            self.cardinality,
            self.dimensionality_hint,
            metadata_bytes,
            permutation_bytes,
            self.name.clone(),
            center_map_bytes,
            leaf_bytes,
        );
        let bytes = bitcode::encode(&members).map_err(|e| e.to_string())?;
        std::fs::write(path, &bytes).map_err(|e| e.to_string())
    }

    fn read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;

        #[allow(clippy::type_complexity)]
        let (cardinality, dimensionality_hint, metadata_bytes, permutation_bytes, name, center_map_bytes, leaf_bytes): (
            usize,
            (usize, Option<usize>),
            Vec<u8>,
            Vec<u8>,
            String,
            Vec<u8>,
            Vec<u8>,
        ) = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

        let metadata: Vec<Me> = decompress_and_decode(&metadata_bytes)?;
        let permutation: Vec<usize> = decompress_and_decode(&permutation_bytes)?;

        let center_map: Vec<(usize, Box<[u8]>)> = decompress_and_decode(&center_map_bytes)?;
        let center_map = center_map
            .into_iter()
            .map(|(i, bytes)| (i, I::from_bytes(&bytes)))
            .collect();

        let leaf_bytes = decompress_and_decode(&leaf_bytes)?;

        Ok(Self {
            cardinality,
            dimensionality_hint,
            metadata,
            permutation,
            name,
            center_map,
            leaf_bytes,
        })
    }
}

#[cfg(feature = "disk-io")]
impl<
        I: Decodable + bitcode::Encode + bitcode::Decode + Send + Sync,
        Me: bitcode::Encode + bitcode::Decode + Send + Sync,
    > crate::dataset::ParDatasetIO<I> for CodecData<I, Me>
{
    fn par_write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let center_map = self
            .center_map
            .par_iter()
            .map(|(&i, p)| (i, p.as_bytes()))
            .collect::<Vec<_>>();

        let ((metadata_bytes, center_map_bytes), (permutation_bytes, leaf_bytes)) = rayon::join(
            || {
                rayon::join(
                    || encode_and_compress(&self.metadata),
                    || encode_and_compress(center_map),
                )
            },
            || {
                rayon::join(
                    || encode_and_compress(&self.permutation),
                    || encode_and_compress(&self.leaf_bytes),
                )
            },
        );

        let (metadata_bytes, center_map_bytes, permutation_bytes, leaf_bytes) =
            (metadata_bytes?, center_map_bytes?, permutation_bytes?, leaf_bytes?);

        let members = (
            self.cardinality,
            self.dimensionality_hint,
            metadata_bytes,
            permutation_bytes,
            self.name.clone(),
            center_map_bytes,
            leaf_bytes,
        );

        let bytes = bitcode::encode(&members).map_err(|e| e.to_string())?;
        std::fs::write(path, &bytes).map_err(|e| e.to_string())
    }

    fn par_read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;

        #[allow(clippy::type_complexity)]
        let (cardinality, dimensionality_hint, metadata_bytes, permutation_bytes, name, center_map_bytes, leaf_bytes): (
            usize,
            (usize, Option<usize>),
            Vec<u8>,
            Vec<u8>,
            String,
            Vec<u8>,
            Vec<u8>,
        ) = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

        #[allow(clippy::type_complexity)]
        let ((metadata_bytes, center_map_bytes), (permutation, leaf_bytes)): (
            (Result<Vec<Me>, String>, Result<Vec<(usize, Box<[u8]>)>, String>),
            (Result<Vec<usize>, String>, Result<Vec<(usize, Box<[u8]>)>, String>),
        ) = rayon::join(
            || {
                rayon::join(
                    || decompress_and_decode(&metadata_bytes),
                    || decompress_and_decode(&center_map_bytes),
                )
            },
            || {
                rayon::join(
                    || decompress_and_decode(&permutation_bytes),
                    || decompress_and_decode(&leaf_bytes),
                )
            },
        );

        let (metadata, center_map_bytes, permutation, leaf_bytes) =
            (metadata_bytes?, center_map_bytes?, permutation?, leaf_bytes?);

        let center_map = center_map_bytes
            .into_par_iter()
            .map(|(i, bytes)| (i, I::from_bytes(&bytes)))
            .collect();

        Ok(Self {
            cardinality,
            dimensionality_hint,
            metadata,
            permutation,
            name,
            center_map,
            leaf_bytes,
        })
    }
}

impl<I: Encodable + Decodable, Me> Decompressible<I> for CodecData<I, Me> {
    fn centers(&self) -> &HashMap<usize, I> {
        &self.center_map
    }

    fn leaf_bytes(&self) -> &[(usize, Box<[u8]>)] {
        &self.leaf_bytes
    }
}

impl<I: Encodable + Decodable + Send + Sync, Me: Send + Sync> ParDecompressible<I> for CodecData<I, Me> {}
