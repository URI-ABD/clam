//! An extension trait for `Dataset` that provides encoding and decoding methods for metrics.

use std::ops::Index;

use distances::{
    number::UInt,
    strings::{
        needleman_wunsch::apply_edits, needleman_wunsch::compute_table, needleman_wunsch::trace_back_recursive,
        needleman_wunsch::unaligned_x_to_y, needleman_wunsch::Edit, Penalties,
    },
    Number,
};

use crate::{Dataset, Instance, VecDataset};

/// An extension trait for `Dataset` that provides encoding and decoding methods for metrics.
#[allow(clippy::module_name_repetitions)]
pub trait SquishyDataset<I: Instance, U: Number>: Dataset<I, U> {
    /// Encodes an instance in the dataset into a byte array using a reference instance.
    fn encode_instance(&self, reference: &I, target: &I) -> Box<[u8]>;

    /// Decodes an instance from a byte array using a reference instance.
    fn decode_instance(&self, reference: &I, encoding: &[u8]) -> I;

    /// Returns the number of bytes required to encode an instance in terms of a reference instance.
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
    fn save(&self, path: &std::path::Path) -> Result<(), String>;

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
        encoder: fn(&String, &String) -> Box<[u8]>,
        decoder: fn(&String, &[u8]) -> String,
    ) -> Result<Self, String>
    where
        Self: Sized;
}

/// A dataset that stores genomic data, and has encoding and decoding methods for the metric involved.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct GenomicDataset<U: UInt> {
    /// The base dataset.
    base_data: VecDataset<String, U, String>,
    /// The number of bytes required to encode an instance in terms of a reference instance.
    bytes_per_unit_distance: u64,
    /// The encoding function.
    encoder: fn(&String, &String) -> Box<[u8]>,
    /// The decoding function.
    decoder: fn(&String, &[u8]) -> String,
}

impl<U: UInt> SquishyDataset<String, U> for GenomicDataset<U> {
    fn encode_instance(&self, reference: &String, target: &String) -> Box<[u8]> {
        (self.encoder)(reference, target)
    }

    fn decode_instance(&self, reference: &String, encoding: &[u8]) -> String {
        (self.decoder)(reference, encoding)
    }

    fn bytes_per_unit_distance(&self) -> u64 {
        self.bytes_per_unit_distance
    }

    #[allow(unused_variables)]
    fn save(&self, path: &std::path::Path) -> Result<(), String> {
        todo!()
    }

    #[allow(unused_variables)]
    fn load(
        path: &std::path::Path,
        metric: fn(&String, &String) -> U,
        is_expensive: bool,
        encoder: fn(&String, &String) -> Box<[u8]>,
        decoder: fn(&String, &[u8]) -> String,
    ) -> Result<Self, String>
    where
        Self: Sized,
    {
        todo!()
    }
}

impl<U: UInt> Dataset<String, U> for GenomicDataset<U> {
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

    #[allow(unused_variables)]
    fn save(&self, path: &std::path::Path) -> Result<(), String> {
        Err("Use `SqishyDataset::save` instead".to_string())
    }

    #[allow(unused_variables)]
    fn load(path: &std::path::Path, metric: fn(&String, &String) -> U, is_expensive: bool) -> Result<Self, String>
    where
        Self: Sized,
    {
        Err("Use `SqishyDataset::load` instead".to_string())
    }
}

impl<U: UInt> Index<usize> for GenomicDataset<U> {
    type Output = String;

    fn index(&self, index: usize) -> &Self::Output {
        &self.base_data[index]
    }
}

#[allow(dead_code)]
#[allow(clippy::unwrap_used)]
/// Encodes a reference and target string into a byte array.
///
/// # Arguments
///
/// * `reference`: The reference string.
/// * `target`: The target string.
///
/// # Returns
///
/// A byte array encoding the reference and target strings.
pub fn encode_general<U: UInt>(reference: &str, target: &str) -> Box<[u8]> {
    let table = compute_table::<U>(reference, target, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_recursive(&table, [reference, target]);

    let edits = unaligned_x_to_y(&aligned_x, &aligned_y);

    let bytes = bincode::serialize(&edits).unwrap();

    bytes.into_boxed_slice()
}

#[allow(unused_variables)]
#[allow(dead_code)]
#[allow(clippy::unwrap_used)]
/// Decodes a reference string from a byte array.
///
/// # Arguments
///
/// * `reference`: The reference string.
/// * `encoding`: The byte array encoding the target string.
///
/// # Returns
///
/// The target string.
pub fn decode_general(reference: &str, encoding: &[u8]) -> String {
    let edits: Vec<Edit> = bincode::deserialize(encoding).unwrap();

    apply_edits(reference, &edits)
}
