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
    #[allow(clippy::type_complexity)]
    fn load(
        path: &std::path::Path,
        metric: fn(&String, &String) -> U,
        is_expensive: bool,
        encoder: fn(&String, &String) -> Result<Box<[u8]>, String>,
        decoder: fn(&String, &[u8]) -> Result<String, String>,
    ) -> Result<Self, String>
    where
        Self: Sized;
}

/// A dataset that stores genomic data, and has encoding and decoding methods for the metric involved.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
#[allow(clippy::type_complexity)]
pub struct GenomicDataset<U: UInt, M: Instance> {
    /// The base dataset.
    pub base_data: VecDataset<String, U, M>,
    /// The number of bytes required to encode an instance in terms of a reference instance.
    pub bytes_per_unit_distance: u64,
    /// The encoding function.
    pub encoder: fn(&String, &String) -> Result<Box<[u8]>, String>,
    /// The decoding function.
    pub decoder: fn(&String, &[u8]) -> Result<String, String>,
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

    #[allow(unused_variables)]
    fn save(&self, path: &std::path::Path) -> Result<(), String> {
        todo!()
    }

    #[allow(unused_variables)]
    fn load(
        path: &std::path::Path,
        metric: fn(&String, &String) -> U,
        is_expensive: bool,
        encoder: fn(&String, &String) -> Result<Box<[u8]>, String>,
        decoder: fn(&String, &[u8]) -> Result<String, String>,
    ) -> Result<Self, String>
    where
        Self: Sized,
    {
        todo!()
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
/// # Returns
///
/// A byte array encoding the reference and target strings.
#[allow(dead_code, clippy::ptr_arg)]
pub fn encode_general<U: UInt>(reference: &String, target: &String) -> Result<Box<[u8]>, String> {
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
/// # Returns
///
/// The target string.
#[allow(dead_code, clippy::ptr_arg)]
pub fn decode_general(reference: &String, encoding: &[u8]) -> Result<String, String> {
    let edits: Vec<Edit> = bincode::deserialize(encoding).map_err(|e| e.to_string())?;

    Ok(apply_edits(reference, &edits))
}

#[cfg(test)]
mod tests {
    use distances::strings::levenshtein;

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
}
