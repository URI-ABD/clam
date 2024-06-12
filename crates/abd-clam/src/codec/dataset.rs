//! An extension trait for `Dataset` that provides encoding and decoding methods for metrics.

use std::{fs::File, io::BufWriter, ops::Index};

use distances::{
    number::UInt,
    strings::{
        needleman_wunsch::{apply_edits, compute_table, trace_back_recursive, unaligned_x_to_y, Edit},
        Penalties,
    },
    Number,
};

use crate::{Cluster, Dataset, Instance, VecDataset};

use super::{DecoderFn, EncoderFn, SquishyBall};

/// An extension trait for `Dataset` that provides encoding and decoding methods for metrics.
#[allow(clippy::module_name_repetitions)]
pub trait SquishyDataset<I: Instance, U: UInt>: Dataset<I, U> {
    /// Encodes an instance in the dataset into a byte array using a reference instance.
    ///
    /// # Errors
    ///
    /// * If the instance cannot be encoded into the given encoding.
    fn encode_instance(&self, reference: &I, target: &I) -> Result<Box<[u8]>, String>;

    /// Decodes an instance from a byte array using a reference instance.
    ///
    /// # Errors
    ///
    /// * If the instance cannot be decoded from the given encoding.
    fn decode_instance(&self, reference: &I, encoding: &[u8]) -> Result<I, String>;

    /// Returns the number of bytes required to encode one edit operation.
    ///
    /// Ideally, this should be proportional to the distance between the two instances.
    ///
    /// This may be a user given estimate.
    fn bytes_per_unit_distance(&self) -> u64;

    /// Saves a `SquishyDataset` to a file.
    ///
    /// This method should compress the dataset and save it to a file.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be saved to the given path.
    fn save(&self, path: &std::path::Path, root: &SquishyBall<U>) -> Result<(), String>;

    /// Loads a `SquishyDataset` from a file.
    ///
    /// This method should decompress the dataset and load it from a file.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be loaded from the given path.
    /// * If the dataset is not the same type as the one that was saved.
    /// * If the file was corrupted.
    fn load(
        path: &std::path::Path,
        metric: fn(&String, &String) -> U,
        is_expensive: bool,
        encoder: EncoderFn<String>,
        decoder: DecoderFn<String>,
        root: &SquishyBall<U>,
    ) -> Result<Self, String>
    where
        Self: Sized;
}

/// A dataset that stores genomic data, and has encoding and decoding methods for the metric involved.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct GenomicDataset<U: UInt, M: Instance> {
    /// The base dataset.
    pub base_data: VecDataset<String, U, M>,
    /// The number of bytes required to encode an instance in terms of a reference instance.
    pub bytes_per_unit_distance: u64,
    /// The encoding function.
    pub encoder: EncoderFn<String>,
    /// The decoding function.
    pub decoder: DecoderFn<String>,
}

impl<U: UInt, M: Instance> GenomicDataset<U, M> {
    /// Creates a new `GenomicDataset`.
    ///
    /// # Arguments
    ///
    /// * `base_data`: The base dataset.
    /// * `bytes_per_unit_distance`: The number of bytes required to encode an instance in terms of a reference instance.
    /// * `encoder`: The encoding function.
    /// * `decoder`: The decoding function.
    pub fn new(
        base_data: VecDataset<String, U, M>,
        bytes_per_unit_distance: u64,
        encoder: EncoderFn<String>,
        decoder: DecoderFn<String>,
    ) -> Self {
        Self {
            base_data,
            bytes_per_unit_distance,
            encoder,
            decoder,
        }
    }

    /// Returns the base dataset.
    #[must_use]
    pub const fn base_data(&self) -> &VecDataset<String, U, M> {
        &self.base_data
    }

    /// Returns the number of bytes required to encode an instance in terms of a reference instance.
    #[must_use]
    pub const fn bytes_per_unit_distance(&self) -> u64 {
        self.bytes_per_unit_distance
    }

    /// Returns the encoding function.
    #[must_use]
    pub fn encoder(&self) -> EncoderFn<String> {
        self.encoder
    }

    /// Returns the decoding function.
    #[must_use]
    pub fn decoder(&self) -> DecoderFn<String> {
        self.decoder
    }

    /// Sets the bytes per unit distance.
    pub fn set_bytes_per_unit_distance(&mut self, bytes_per_unit_distance: u64) {
        self.bytes_per_unit_distance = bytes_per_unit_distance;
    }

    /// Sets the encoder function.
    pub fn set_encoder(&mut self, encoder: EncoderFn<String>) {
        self.encoder = encoder;
    }

    /// Sets the decoder function.
    pub fn set_decoder(&mut self, decoder: DecoderFn<String>) {
        self.decoder = decoder;
    }

    /// Write the dataset to a writer.
    ///
    /// # Arguments
    ///
    /// * `writer`: The writer to write the dataset to.
    /// * `root`: The root cluster to write.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be written to the writer.
    pub fn save<W>(&self, writer: &mut W, root: &SquishyBall<U>) -> Result<(), String>
    where
        W: ?Sized + std::io::Write,
    {
        // TODO: Move most of this method to the `SquishyDataset` trait instead.

        // Write the header for basic protection against corruption.
        let type_name = Self::type_name();
        writer
            .write_all(&type_name.len().to_le_bytes())
            .and_then(|()| writer.write_all(type_name.as_bytes()))
            .map_err(|e| e.to_string())?;

        // Write the name of the dataset.
        let name = self.base_data.name();
        writer
            .write_all(&(name.len() as u64).to_le_bytes())
            .and_then(|()| writer.write_all(name.as_bytes()))
            .map_err(|e| e.to_string())?;

        // Write the cardinality of the dataset.
        writer
            .write_all(&(self.base_data.cardinality() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Write the permutation of the dataset.
        let permutation = self
            .base_data
            .permuted_indices()
            .map_or(Vec::new(), |p| p.iter().flat_map(|i| i.to_le_bytes()).collect());
        writer
            .write_all(&(permutation.len() as u64).to_le_bytes())
            .and_then(|()| writer.write_all(&permutation))
            .map_err(|e| e.to_string())?;

        // Write the number of metadata.
        writer
            .write_all(&(self.base_data.metadata.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Save the metadata.
        for meta in &self.base_data.metadata {
            meta.save(writer)?;
        }

        // Save the `bytes_per_unit_distance`.
        writer
            .write_all(&self.bytes_per_unit_distance.to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Save the center of the root.
        let center = &self.base_data[root.arg_center()];
        writer
            .write_all(&(center.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer.write_all(center.as_bytes()).map_err(|e| e.to_string())?;

        // Write the base dataset.
        self.save_base(writer, root)
    }

    /// Write the base dataset to a writer.
    ///
    /// # Arguments
    ///
    /// * `writer`: The writer to write the dataset to.
    /// * `c`: The cluster to write.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be written to the writer.
    fn save_base<W>(&self, writer: &mut W, c: &SquishyBall<U>) -> Result<(), String>
    where
        W: ?Sized + std::io::Write,
    {
        if let Some([left, right]) = c.children() {
            // Get the centers of the current cluster and its children.
            let p_center = &self.base_data[c.arg_center()];
            let l_center = &self.base_data[c.arg_center()];
            let r_center = &self.base_data[c.arg_center()];

            // Encode the centers of the children in terms of the center of the parent.
            let l_encoding = self.encode_instance(p_center, l_center)?;
            let r_encoding = self.encode_instance(p_center, r_center)?;

            // Write the header.
            let header = b"Parent";
            writer.write_all(header).map_err(|e| e.to_string())?;

            // Write the length of the encoding followed by the encoding itself for the left child.
            writer
                .write_all(&(l_encoding.len() as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer.write_all(&l_encoding).map_err(|e| e.to_string())?;

            // Write the length of the encoding followed by the encoding itself for the right child.
            writer
                .write_all(&(r_encoding.len() as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer.write_all(&r_encoding).map_err(|e| e.to_string())?;

            // Recursively write the children.
            self.save_base(writer, left)?;
            self.save_base(writer, right)
        } else {
            let center = &self.base_data[c.arg_center()];
            let encodings = c
                .indices()
                .map(|i| self.encode_instance(center, &self.base_data[i]).map(|e| (e.len(), e)))
                .collect::<Result<Vec<_>, _>>()?;

            // Write the header. The header is "Leaf" followed by the the number of encodings.
            let header = b"Leaf";
            writer.write_all(header).map_err(|e| e.to_string())?;
            writer
                .write_all(&(encodings.len() as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;

            // For each encoding, write the length of the encoding followed by the encoding itself.
            for (len, encoding) in encodings {
                writer
                    .write_all(&(len as u64).to_le_bytes())
                    .map_err(|e| e.to_string())?;
                writer.write_all(&encoding).map_err(|e| e.to_string())?;
            }

            Ok(())
        }
    }

    /// Read the dataset from a file.
    ///
    /// # Arguments
    ///
    /// * `reader`: The reader to read the dataset from.
    /// * `root`: The root cluster to read.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be read from the reader.
    ///
    /// # Returns
    ///
    /// The dataset read from the reader.
    pub fn load<R>(
        reader: &mut R,
        metric: fn(&String, &String) -> U,
        is_expensive: bool,
        encoder: EncoderFn<String>,
        decoder: DecoderFn<String>,
        root: &SquishyBall<U>,
    ) -> Result<Self, String>
    where
        R: ?Sized + std::io::Read,
    {
        // TODO: Move most of this method to the `SquishyDataset` trait instead.

        // Read the header and check if it is the same as the type name.
        {
            let mut type_name_len = vec![0u8; usize::num_bytes()];
            reader.read_exact(&mut type_name_len).map_err(|e| e.to_string())?;
            let type_name_len = <usize as Number>::from_le_bytes(&type_name_len);
            let mut type_name = vec![0u8; type_name_len];
            reader.read_exact(&mut type_name).map_err(|e| e.to_string())?;
            let type_name = std::str::from_utf8(&type_name).map_err(|e| e.to_string())?;
            if type_name != Self::type_name() {
                return Err(format!(
                    "Expected type name '{}', got '{}'.",
                    Self::type_name(),
                    type_name
                ));
            }
        }

        // Read the name of the dataset.
        let name = {
            let mut name_len = vec![0u8; usize::num_bytes()];
            reader.read_exact(&mut name_len).map_err(|e| e.to_string())?;
            let name_len = <usize as Number>::from_le_bytes(&name_len);
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf).map_err(|e| e.to_string())?;
            String::from_utf8(name_buf).map_err(|e| e.to_string())?
        };

        // Read the cardinality of the dataset.
        let mut cardinality = vec![0u8; usize::num_bytes()];
        reader.read_exact(&mut cardinality).map_err(|e| e.to_string())?;
        let cardinality = <usize as Number>::from_le_bytes(&cardinality);

        // Read the permutation of the dataset.
        let permutation = {
            let mut permutation_len = vec![0u8; usize::num_bytes()];
            reader.read_exact(&mut permutation_len).map_err(|e| e.to_string())?;
            let permutation_len = <usize as Number>::from_le_bytes(&permutation_len);

            if permutation_len == 0 {
                None
            } else {
                let mut permutation = vec![0u8; 8 * cardinality];
                reader.read_exact(&mut permutation).map_err(|e| e.to_string())?;
                let permutation = permutation
                    .chunks_exact(8)
                    .map(<usize as Number>::from_le_bytes)
                    .collect::<Vec<_>>();
                Some(permutation)
            }
        };

        // Read the number of metadata.
        let mut num_metadata = vec![0u8; usize::num_bytes()];
        reader.read_exact(&mut num_metadata).map_err(|e| e.to_string())?;
        let num_metadata = <usize as Number>::from_le_bytes(&num_metadata);

        // Read the metadata.
        let metadata = (0..num_metadata)
            .map(|_| M::load(reader))
            .collect::<Result<Vec<_>, _>>()?;

        // Read the `bytes_per_unit_distance` field.
        let mut bytes_per_unit_distance = vec![0u8; usize::num_bytes()];
        reader
            .read_exact(&mut bytes_per_unit_distance)
            .map_err(|e| e.to_string())?;
        let bytes_per_unit_distance = <u64 as Number>::from_le_bytes(&bytes_per_unit_distance);

        // Read the center of the root.
        let center = {
            let mut len = vec![0u8; usize::num_bytes()];
            reader.read_exact(&mut len).map_err(|e| e.to_string())?;
            let len = <usize as Number>::from_le_bytes(&len);
            let mut center = vec![0u8; len];
            reader.read_exact(&mut center).map_err(|e| e.to_string())?;
            String::from_utf8(center).map_err(|e| e.to_string())?
        };

        // Read the base dataset.
        let base_data = Self::load_base(reader, root, &center, decoder)?;

        let base_data = VecDataset {
            name,
            data: base_data,
            metric,
            is_expensive,
            permuted_indices: permutation,
            metadata,
        };

        Ok(Self {
            base_data,
            bytes_per_unit_distance,
            encoder,
            decoder,
        })
    }

    /// Read the base dataset from a file.
    ///
    /// # Arguments
    ///
    /// * `reader`: The reader to read the dataset from.
    /// * `c`: The cluster to read.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be read from the reader.
    ///
    /// # Returns
    ///
    /// The `Vec` of decoded instances.
    fn load_base<R>(
        reader: &mut R,
        c: &SquishyBall<U>,
        center: &String,
        decoder: DecoderFn<String>,
    ) -> Result<Vec<String>, String>
    where
        R: ?Sized + std::io::Read,
    {
        if let Some([left, right]) = c.children() {
            // Read the header and check if it is "Parent".
            let mut parent = [0u8; 6];
            reader.read_exact(&mut parent).map_err(|e| e.to_string())?;
            let parent = std::str::from_utf8(&parent).map_err(|e| e.to_string())?;
            if parent != "Parent" {
                return Err(format!("Expected header 'Parent', got '{parent}'."));
            }

            // Read the length of the encoding followed by the encoding itself for the left child.
            let l_len = {
                let mut l_len = vec![0u8; usize::num_bytes()];
                reader.read_exact(&mut l_len).map_err(|e| e.to_string())?;
                <usize as Number>::from_le_bytes(&l_len)
            };
            let l_encoding = {
                let mut l_encoding = vec![0u8; l_len];
                reader.read_exact(&mut l_encoding).map_err(|e| e.to_string())?;
                l_encoding
            };

            // Read the length of the encoding followed by the encoding itself for the right child.
            let r_len = {
                let mut r_len = vec![0u8; usize::num_bytes()];
                reader.read_exact(&mut r_len).map_err(|e| e.to_string())?;
                <usize as Number>::from_le_bytes(&r_len)
            };
            let r_encoding = {
                let mut r_encoding = vec![0u8; r_len];
                reader.read_exact(&mut r_encoding).map_err(|e| e.to_string())?;
                r_encoding
            };

            // Decode the encodings.
            let l_center = decoder(center, &l_encoding)?;
            let r_center = decoder(center, &r_encoding)?;

            // Recursively read the children.
            let mut left_data = Self::load_base(reader, left, &l_center, decoder)?;
            let mut right_data = Self::load_base(reader, right, &r_center, decoder)?;

            left_data.append(&mut right_data);
            Ok(left_data)
        } else {
            // Read the header and check if it is "Leaf".
            let mut leaf = [0u8; 4];
            reader.read_exact(&mut leaf).map_err(|e| e.to_string())?;
            let leaf = std::str::from_utf8(&leaf).map_err(|e| e.to_string())?;
            if leaf != "Leaf" {
                return Err(format!("Expected header 'Leaf', got '{leaf}'."));
            }

            // Read the number of encodings.
            let num_encodings = {
                let mut num_encodings = vec![0u8; usize::num_bytes()];
                reader.read_exact(&mut num_encodings).map_err(|e| e.to_string())?;
                <usize as Number>::from_le_bytes(&num_encodings)
            };

            // Read the encodings.
            let mut data = Vec::with_capacity(num_encodings);
            for _ in 0..num_encodings {
                let len = {
                    let mut len = vec![0u8; usize::num_bytes()];
                    reader.read_exact(&mut len).map_err(|e| e.to_string())?;
                    <usize as Number>::from_le_bytes(&len)
                };
                let mut encoding = vec![0u8; len];
                reader.read_exact(&mut encoding).map_err(|e| e.to_string())?;
                let instance = decoder(center, &encoding)?;
                data.push(instance);
            }

            Ok(data)
        }
    }
}

impl<U: UInt, M: Instance> SquishyDataset<String, U> for GenomicDataset<U, M> {
    fn encode_instance(&self, reference: &String, target: &String) -> Result<Box<[u8]>, String> {
        (self.encoder)(reference, target)
    }

    fn decode_instance(&self, reference: &String, encoding: &[u8]) -> Result<String, String> {
        (self.decoder)(reference, encoding)
    }

    fn bytes_per_unit_distance(&self) -> u64 {
        self.bytes_per_unit_distance
    }

    fn save(&self, path: &std::path::Path, root: &SquishyBall<U>) -> Result<(), String> {
        // Check that the path does not exist but its parent does.
        if path.exists() {
            return Err(format!("Path '{}' already exists.", path.display()));
        }

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                return Err(format!("Parent path '{}' does not exist.", parent.display()));
            }
        }

        // Save the dataset.
        let mut handle = File::create(path).map_err(|e| e.to_string())?;
        self.save(&mut handle, root)
    }

    fn load(
        path: &std::path::Path,
        metric: fn(&String, &String) -> U,
        is_expensive: bool,
        encoder: fn(&String, &String) -> Result<Box<[u8]>, String>,
        decoder: fn(&String, &[u8]) -> Result<String, String>,
        root: &SquishyBall<U>,
    ) -> Result<Self, String>
    where
        Self: Sized,
    {
        let mut handle = File::open(path.join("dataset")).map_err(|e| e.to_string())?;
        Self::load(&mut handle, metric, is_expensive, encoder, decoder, root)
    }
}

impl<U: UInt, M: Instance> Dataset<String, U> for GenomicDataset<U, M> {
    fn type_name() -> String {
        format!("GenomicDataset<{}>", U::type_name())
    }

    fn name(&self) -> &str {
        self.base_data.name()
    }

    fn cardinality(&self) -> usize {
        self.base_data.cardinality()
    }

    fn is_metric_expensive(&self) -> bool {
        self.base_data.is_metric_expensive()
    }

    fn metric(&self) -> fn(&String, &String) -> U {
        self.base_data.metric()
    }

    fn set_permuted_indices(&mut self, indices: Option<&[usize]>) {
        self.base_data.set_permuted_indices(indices);
    }

    fn swap(&mut self, left: usize, right: usize) -> Result<(), String> {
        self.base_data.swap(left, right)
    }

    fn permuted_indices(&self) -> Option<&[usize]> {
        self.base_data.permuted_indices()
    }

    fn make_shards(self, max_cardinality: usize) -> Vec<Self>
    where
        Self: Sized,
    {
        let base_shards = self.base_data.make_shards(max_cardinality);
        base_shards
            .into_iter()
            .map(|base_data| Self {
                base_data,
                bytes_per_unit_distance: self.bytes_per_unit_distance,
                encoder: self.encoder,
                decoder: self.decoder,
            })
            .collect()
    }

    fn save(&self, path: &std::path::Path) -> Result<(), String> {
        // Check that the path does not exist but its parent does.
        if path.exists() {
            return Err(format!("Path '{}' already exists.", path.display()));
        }

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                return Err(format!("Parent path '{}' does not exist.", parent.display()));
            }
        }

        // Create a directory at the path for the dataset.
        std::fs::create_dir(path).map_err(|e| e.to_string())?;

        // Save the base dataset.
        let base_path = path.join("base");
        self.base_data.save(&base_path)?;

        // Save the `bytes_per_unit_distance` field in a binary file.
        let bytes_per_unit_distance_path = path.join("bytes_per_unit_distance");
        let mut writer = BufWriter::new(File::create(bytes_per_unit_distance_path).map_err(|e| e.to_string())?);
        bincode::serialize_into(&mut writer, &self.bytes_per_unit_distance).map_err(|e| e.to_string())?;

        Ok(())
    }

    fn load(path: &std::path::Path, metric: fn(&String, &String) -> U, is_expensive: bool) -> Result<Self, String>
    where
        Self: Sized,
    {
        // Load the base dataset.
        let base_path = path.join("base");
        let base_data = VecDataset::load(&base_path, metric, is_expensive)?;

        // Load the `bytes_per_unit_distance` field from a binary file.
        let bytes_per_unit_distance_path = path.join("bytes_per_unit_distance");
        let bytes_per_unit_distance = {
            let reader = File::open(bytes_per_unit_distance_path).map_err(|e| e.to_string())?;
            bincode::deserialize_from(reader).map_err(|e| e.to_string())?
        };

        Ok(Self {
            base_data,
            bytes_per_unit_distance,
            encoder: encode_general::<U>,
            decoder: decode_general,
        })
    }
}

impl<U: UInt, M: Instance> Index<usize> for GenomicDataset<U, M> {
    type Output = String;

    fn index(&self, index: usize) -> &Self::Output {
        &self.base_data[index]
    }
}

/// Encodes a reference and target string into a byte array.
///
/// # Arguments
///
/// * `reference`: The reference string.
/// * `target`: The target string.
///
/// # Errors
///
/// * If the byte array cannot be serialized.
///
/// # Returns
///
/// A byte array encoding the reference and target strings.
#[allow(dead_code, clippy::ptr_arg)]
pub fn encode_general<U: UInt>(reference: &String, target: &String) -> Result<Box<[u8]>, String> {
    // TODO(Morgan): Correct the Errors section in docs.
    let table = compute_table::<U>(reference, target, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_recursive(&table, [reference, target]);

    let edits = unaligned_x_to_y(&aligned_x, &aligned_y);

    let bytes = bincode::serialize(&edits).map_err(|e| e.to_string())?;

    Ok(bytes.into_boxed_slice())
}

/// Decodes a reference string from a byte array.
///
/// # Arguments
///
/// * `reference`: The reference string.
/// * `encoding`: The byte array encoding the target string.
///
/// # Errors
///
/// * If the byte array cannot be deserialized into a vector of edits.
///
/// # Returns
///
/// The target string.
#[allow(dead_code, clippy::ptr_arg)]
pub fn decode_general(reference: &String, encoding: &[u8]) -> Result<String, String> {
    // TODO(Morgan): Correct the Errors section in docs.
    let edits: Vec<Edit> = bincode::deserialize(encoding).map_err(|e| e.to_string())?;

    Ok(apply_edits(reference, &edits))
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Write};

    use distances::strings::levenshtein;
    use tempdir::TempDir;

    use crate::{codec::criteria::CompressionCriteria, PartitionCriteria, UniBall};

    use super::*;

    fn lev_metric(x: &String, y: &String) -> u16 {
        levenshtein(x, y)
    }

    #[test]
    fn test_encode_decode() {
        let reference = "NAJIBPEPPERSEATS".to_string();
        let target = "NAJIBEATSPEPPERS".to_string();

        let encoding = encode_general::<u8>(&reference, &target).unwrap();
        let decoded = decode_general(&reference, &encoding).unwrap();

        assert_eq!(decoded, target);
    }

    #[test]
    fn test_genomic() {
        let strings = vec![
            "NAJIBPEPPERSEATS".to_string(),
            "NAJIBEATSPEPPERS".to_string(),
            "TOMEATSWHATFOODEATS".to_string(),
            "FOODEATSWHATTOMEATS".to_string(),
        ];

        let base_data = VecDataset::new("test-genomic".to_string(), strings.clone(), lev_metric, true);

        let dataset = GenomicDataset {
            base_data,
            bytes_per_unit_distance: 1,
            encoder: encode_general::<u16>,
            decoder: decode_general,
        };

        for i in 0..strings.len() {
            for j in 0..strings.len() {
                let encoding = dataset.encode_instance(&strings[i], &strings[j]).unwrap();
                if i == j {
                    let edits = bincode::deserialize::<Vec<Edit>>(&encoding).unwrap();
                    assert!(edits.is_empty());
                } else {
                    let decoded = dataset.decode_instance(&strings[i], &encoding).unwrap();
                    assert_eq!(decoded, strings[j]);
                }
            }
        }
    }

    #[test]
    fn test_save_load() {
        let strings = vec![
            "NAJIBPEPPERSEATS".to_string(),
            "NAJIBEATSPEPPERS".to_string(),
            "TOMEATSWHATFOODEATS".to_string(),
            "FOODEATSWHATTOMEATS".to_string(),
        ];

        let base_data = VecDataset::new("test-genomic".to_string(), strings.clone(), lev_metric, true);

        let mut data = GenomicDataset {
            base_data,
            bytes_per_unit_distance: 1,
            encoder: encode_general::<u16>,
            decoder: decode_general,
        };

        let seed = Some(42);
        let criteria = PartitionCriteria::default();
        let root = UniBall::new_root(&data, seed).partition(&mut data, &criteria, seed);
        let root = {
            let mut root = SquishyBall::from_base_tree(root);
            let criteria = CompressionCriteria::new(false).with_fixed_depth(2);
            root.apply_criteria(&criteria);
            root
        };

        // Create a temporary directory for the dataset.
        let temp_dir = TempDir::new("test-genomic").unwrap();
        let temp_path = temp_dir.path().join("dataset");

        // Save the dataset.
        let mut writer = BufWriter::new(File::create(&temp_path).unwrap());
        data.save(&mut writer, &root).unwrap();
        writer.flush().unwrap();

        // Load the dataset.
        let mut reader = BufReader::new(File::open(&temp_path).unwrap());
        let loaded = GenomicDataset::<u16, usize>::load(
            &mut reader,
            lev_metric,
            true,
            encode_general::<u16>,
            decode_general,
            &root,
        )
        .unwrap();

        // Check that the loaded dataset is the same as the original.
        assert_eq!(data.name(), loaded.name());
        assert_eq!(data.base_data.name(), loaded.base_data.name());
        assert_eq!(data.base_data.cardinality(), loaded.base_data.cardinality());
        assert_eq!(data.base_data.metadata, loaded.base_data.metadata);
        assert_eq!(data.bytes_per_unit_distance, loaded.bytes_per_unit_distance);
        assert_eq!(data.base_data.permuted_indices(), loaded.base_data.permuted_indices());
    }
}
