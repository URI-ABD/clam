//! A dataset for searching in a compressed space.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use distances::{number::UInt, Number};

use crate::{Cluster, Dataset, Instance};

use super::{DecoderFn, EncoderFn, SquishyBall};

/// A `Dataset` that allows for searching in a compressed space.
#[derive(Debug)]
pub struct CodecData<I: Instance, U: UInt, M: Instance> {
    /// The root of the squishy ball tree.
    root: SquishyBall<U>,
    /// The subset of the dataset that contains the centers of the clusters in the tree.
    centers: HashMap<usize, I>,
    /// The encoding function.
    encoder: EncoderFn<I>,
    /// The compressed data for the squished clusters.
    leaf_data: LeafData<I>,
    /// The distance function.
    metric: fn(&I, &I) -> U,
    /// Whether the distance function is expensive to compute.
    is_expensive: bool,
    /// Metadata for the dataset.
    metadata: Vec<M>,
    /// The reordering of the dataset after building the tree.
    permuted_indices: Vec<usize>,
}

impl<I: Instance, U: UInt, M: Instance> CodecData<I, U, M> {
    /// Creates a new `CodecData`.
    ///
    /// # Arguments
    ///
    /// * `root`: The root of the squishy ball tree.
    /// * `data`: The dataset to compress.
    /// * `encoder`: The encoding function.
    /// * `decoder`: The decoding function.
    /// * `metadata`: Metadata for the dataset.
    ///
    /// # Returns
    ///
    /// The `CodecData` for the dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if any leaf data could not be encoded.
    pub fn new<D: Dataset<I, U>>(
        mut root: SquishyBall<U>,
        data: &D,
        encoder: EncoderFn<I>,
        decoder: DecoderFn<I>,
        metadata: Vec<M>,
    ) -> Result<Self, String> {
        let permuted_indices = data
            .permuted_indices()
            .map_or_else(|| (0..data.cardinality()).collect(), <[usize]>::to_vec);

        // Trim the tree.
        root.trim();

        // Build the centers
        let subtree = root.compressible_subtree();
        let centers = subtree.iter().map(|c| c.arg_center()).collect::<HashSet<_>>();
        let centers = centers
            .into_iter()
            .map(|i| (i, data[i].clone()))
            .collect::<HashMap<_, _>>();

        // Build the leaves' data
        let mut bytes = Vec::new();
        for leaf in root.compressible_leaves_mut().into_iter().filter(|c| c.squish()) {
            // Set the offset in the compressed data where the leaf data starts.
            leaf.set_codec_offset(bytes.len());

            // Encode the points in the leaf in terms of the center.
            let center = &data[leaf.arg_center()];
            let encodings = leaf
                .indices()
                .map(|i| encoder(center, &data[i]))
                .collect::<Result<Vec<_>, _>>()?;

            // Write the number of encodings.
            bytes.extend_from_slice(&leaf.cardinality().to_le_bytes());
            for encoding in encodings {
                // Write the length of the encoding.
                let len = encoding.len();
                bytes.extend_from_slice(&len.to_le_bytes());
                // Write the encoding.
                bytes.extend_from_slice(&encoding);
            }
        }

        let bytes = bytes.into_boxed_slice();
        let leaf_data = LeafData { bytes, decoder };

        Ok(Self {
            root,
            centers,
            encoder,
            leaf_data,
            metric: data.metric(),
            is_expensive: data.is_metric_expensive(),
            metadata,
            permuted_indices,
        })
    }

    /// Loads the data for a leaf.
    ///
    /// # Arguments
    ///
    /// * `leaf`: The leaf to load.
    ///
    /// # Returns
    ///
    /// The data for the leaf.
    ///
    /// # Errors
    ///
    /// Returns an error if any leaf data could not be decoded.
    pub fn load_leaf_data(&self, leaf: &SquishyBall<U>) -> Result<Vec<I>, String> {
        let offset = leaf.codec_offset().ok_or("Leaf has no codec offset")?;
        let center = &self.centers[&leaf.arg_center()];
        self.leaf_data.load_leaf(center, offset)
    }

    /// Returns the root of the squishy ball tree.
    pub const fn root(&self) -> &SquishyBall<U> {
        &self.root
    }

    /// Returns the centers of the clusters in the tree.
    pub const fn centers(&self) -> &HashMap<usize, I> {
        &self.centers
    }

    /// Returns the metadata for the dataset.
    pub fn metadata(&self) -> &[M] {
        &self.metadata
    }

    /// Returns the reordering of the dataset after building the tree.
    pub fn permuted_indices(&self) -> &[usize] {
        &self.permuted_indices
    }

    /// Returns the distance function.
    pub fn metric(&self) -> fn(&I, &I) -> U {
        self.metric
    }

    /// Returns whether the distance function is expensive to compute.
    pub const fn is_expensive(&self) -> bool {
        self.is_expensive
    }

    /// Saves the `CodecData` to disk.
    ///
    /// # Arguments
    ///
    /// * `path`: The directory to save the `CodecData`.
    ///
    /// # Errors
    ///
    /// * If the `path`s parent directory does not exist.
    /// * If lacking permissions to write to the `path`.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        // Check if the parent directory exists.
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                return Err(format!("Parent directory does not exist: {parent:?}"));
            }
        } else {
            return Err("Path has no parent directory".to_string());
        }

        // If the directory already exists, delete it.
        if path.exists() {
            std::fs::remove_dir_all(path).map_err(|e| e.to_string())?;
        }

        // Create the directory.
        std::fs::create_dir(path).map_err(|e| e.to_string())?;

        // Save the root.
        let root_path = path.join("root.bin");
        self.root.save(&root_path)?;

        // Save the centers.
        let centers_path = path.join("centers.bin");
        let centers = encode_centers(&self.root, &self.centers, self.encoder)?;
        std::fs::write(centers_path, &centers).map_err(|e| e.to_string())?;

        // Save the leaf data.
        let leaf_data_path = path.join("leaf_data.bin");
        std::fs::write(leaf_data_path, &self.leaf_data.bytes).map_err(|e| e.to_string())?;

        // Save the metadata.
        let metadata_path = path.join("metadata.bin");
        let metadata = self.metadata.iter().map(M::to_bytes).collect::<Vec<_>>();
        let metadata = bincode::serialize(&metadata).map_err(|e| e.to_string())?;
        std::fs::write(metadata_path, metadata).map_err(|e| e.to_string())?;

        // Save the permuted indices.
        let permuted_indices_path = path.join("permuted_indices.bin");
        let permuted_indices = bincode::serialize(&self.permuted_indices).map_err(|e| e.to_string())?;
        std::fs::write(permuted_indices_path, permuted_indices).map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Loads the `CodecData` from disk.\
    ///
    /// # Arguments
    ///
    /// * `path`: The directory where the `CodecData` is saved.
    /// * `metric`: The distance function.
    /// * `is_expensive`: Whether the distance function is expensive to compute.
    /// * `encoder`: The encoding function.
    /// * `decoder`: The decoding function.
    ///
    /// # Returns
    ///
    /// The `CodecData` loaded from disk.
    ///
    /// # Errors
    ///
    /// * If the directory does not exist.
    /// * If the path is not a directory.
    /// * If any of the files do not exist.
    /// * If any of the files cannot be read.
    /// * If any of the files cannot be deserialized.
    pub fn load(
        path: &Path,
        metric: fn(&I, &I) -> U,
        is_expensive: bool,
        encoder: EncoderFn<I>,
        decoder: DecoderFn<I>,
    ) -> Result<Self, String> {
        // Check if the directory exists.
        if !path.exists() {
            return Err(format!("Directory does not exist: {path:?}"));
        }

        // Check if the path is a directory.
        if !path.is_dir() {
            return Err(format!("Path is not a directory: {path:?}"));
        }

        // Check if all the files exist.
        let root_path = path.join("root.bin");
        let centers_path = path.join("centers.bin");
        let leaf_data_path = path.join("leaf_data.bin");
        let metadata_path = path.join("metadata.bin");
        let permuted_indices_path = path.join("permuted_indices.bin");

        for file in [
            &root_path,
            &centers_path,
            &leaf_data_path,
            &metadata_path,
            &permuted_indices_path,
        ] {
            if !file.exists() {
                return Err(format!("File does not exist: {file:?}"));
            }
        }

        // Load the root.
        let root = SquishyBall::load(&root_path)?;

        // Load the centers.
        let centers = std::fs::read(&centers_path).map_err(|e| e.to_string())?;
        let centers = decode_centers(&root, &centers, decoder)?;

        // Load the leaf data.
        let leaf_data = std::fs::read(&leaf_data_path).map_err(|e| e.to_string())?;
        let leaf_data = LeafData {
            bytes: leaf_data.into_boxed_slice(),
            decoder,
        };

        // Load the metadata.
        let metadata = std::fs::read(&metadata_path).map_err(|e| e.to_string())?;
        let metadata: Vec<Vec<u8>> = bincode::deserialize(&metadata).map_err(|e| e.to_string())?;
        let metadata = metadata
            .into_iter()
            .map(|m| M::from_bytes(&m))
            .collect::<Result<Vec<_>, _>>()?;

        // Load the permuted indices.
        let permuted_indices = std::fs::read(&permuted_indices_path).map_err(|e| e.to_string())?;
        let permuted_indices = bincode::deserialize(&permuted_indices).map_err(|e| e.to_string())?;

        Ok(Self {
            root,
            centers,
            encoder,
            leaf_data,
            metric,
            is_expensive,
            metadata,
            permuted_indices,
        })
    }
}

/// Recursively encodes the `centers`.
fn encode_centers<I: Instance, U: UInt>(
    root: &SquishyBall<U>,
    centers: &HashMap<usize, I>,
    encoder: EncoderFn<I>,
) -> Result<Box<[u8]>, String> {
    // Create a buffer to store the bytes.
    let mut bytes = Vec::new();

    // Serialize the root center.
    let root_center = centers[&root.arg_center()].to_bytes();
    bytes.extend_from_slice(&root_center.len().to_le_bytes());
    bytes.extend_from_slice(&root_center);

    // Serialize the index pairs and their encodings.
    for (reference, target) in index_pairs(root) {
        bytes.extend_from_slice(&reference.to_le_bytes());
        bytes.extend_from_slice(&target.to_le_bytes());

        let encoding = encoder(&centers[&reference], &centers[&target])?;

        bytes.extend_from_slice(&encoding.len().to_le_bytes());
        bytes.extend_from_slice(&encoding);
    }

    Ok(bytes.into_boxed_slice())
}

/// Recursively finds the index pairs in the tree that need to be encoded in terms of each other.
///
/// # Arguments
///
/// * `c`: The current node in the tree.
///
/// # Returns
///
/// The index pairs that need to be encoded in terms of each other.
fn index_pairs<U: UInt>(c: &SquishyBall<U>) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    if !c.squish() {
        if let Some([left, right]) = c.children() {
            pairs.push((c.arg_center(), left.arg_center()));
            pairs.push((c.arg_center(), right.arg_center()));
            pairs.append(&mut index_pairs(left));
            pairs.append(&mut index_pairs(right));
        }
    }
    pairs
}

/// Recursively decodes the `centers`.
fn decode_centers<I: Instance, U: UInt>(
    root: &SquishyBall<U>,
    bytes: &[u8],
    decoder: DecoderFn<I>,
) -> Result<HashMap<usize, I>, String> {
    let mut centers = HashMap::new();
    let mut offset = 0;

    // Deserialize the root center.
    let len = <usize as Number>::from_le_bytes(&bytes[offset..(offset + usize::num_bytes())]);
    offset += usize::num_bytes();
    let root_center = I::from_bytes(&bytes[offset..(offset + len)])?;
    centers.insert(root.arg_center(), root_center);
    offset += len;

    // Deserialize the index pairs and their encodings.
    while offset < bytes.len() {
        // Deserialize the index pair.
        let reference = <usize as Number>::from_le_bytes(&bytes[offset..(offset + usize::num_bytes())]);
        offset += usize::num_bytes();
        let target = <usize as Number>::from_le_bytes(&bytes[offset..(offset + usize::num_bytes())]);
        offset += usize::num_bytes();

        // Deserialize the encoding.
        let len = <usize as Number>::from_le_bytes(&bytes[offset..(offset + usize::num_bytes())]);
        offset += usize::num_bytes();
        let encoding = decoder(&centers[&reference], &bytes[offset..(offset + len)])?;
        centers.insert(target, encoding);
        offset += len;
    }

    Ok(centers)
}

/// The compressed data for the squished clusters.
#[derive(Debug)]
struct LeafData<I: Instance> {
    /// The compressed data for the squished clusters.
    pub bytes: Box<[u8]>,
    /// The decoding function.
    pub decoder: DecoderFn<I>,
}

impl<I: Instance> LeafData<I> {
    /// Loads the data for a leaf.
    ///
    /// # Arguments
    ///
    /// * `center`: The center of the leaf.
    /// * `offset`: The offset in the compressed data where the leaf data starts.
    ///
    /// # Returns
    ///
    /// The data for the leaf.
    ///
    /// # Errors
    ///
    /// Returns an error if any leaf data could not be decoded.
    fn load_leaf(&self, center: &I, offset: usize) -> Result<Vec<I>, String> {
        // Read the number of encodings.
        let cardinality = {
            let bytes = &self.bytes[offset..(offset + usize::num_bytes())];
            <usize as Number>::from_le_bytes(bytes)
        };

        let mut data = Vec::with_capacity(cardinality);
        let mut offset = offset + usize::num_bytes();
        for _ in 0..cardinality {
            // Read the length of the encoding.
            let len = {
                let bytes = &self.bytes[offset..(offset + usize::num_bytes())];
                <usize as Number>::from_le_bytes(bytes)
            };
            offset += usize::num_bytes();

            // Read the encoding.
            let target = {
                let bytes = &self.bytes[offset..(offset + len)];
                offset += len;
                (self.decoder)(center, bytes)?
            };

            data.push(target);
        }

        Ok(data)
    }
}
