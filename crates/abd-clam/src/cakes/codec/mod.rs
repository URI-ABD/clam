//! Compression and Decompression with CLAM

mod codec_data;
mod compression;
mod decompression;
mod squishy_ball;

use distances::Number;

#[allow(clippy::module_name_repetitions)]
pub use codec_data::CodecData;
pub use compression::{Compressible, Encodable};
pub use decompression::{Decodable, Decompressible};
pub use squishy_ball::SquishyBall;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, FlatVec};

impl<I: Encodable, U: Number, M> Compressible<I, U> for FlatVec<I, U, M> {}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    super::cluster::Searchable<I, U, Dec> for SquishyBall<I, U, Co, Dec, S>
{
}

impl<
        I: Encodable + Decodable + Send + Sync,
        U: Number,
        Co: Compressible<I, U> + ParDataset<I, U>,
        Dec: Decompressible<I, U> + ParDataset<I, U>,
        S: ParCluster<I, U, Co>,
    > super::cluster::ParSearchable<I, U, Dec> for SquishyBall<I, U, Co, Dec, S>
{
}

/// Reads an encoded value from a byte array and increments the offset.
pub fn read_encoding(bytes: &[u8], offset: &mut usize) -> Box<[u8]> {
    let len = read_usize(bytes, offset);
    let encoding = bytes[*offset..*offset + len].to_vec();
    *offset += len;
    encoding.into_boxed_slice()
}

/// Reads a `usize` from a byte array and increments the offset.
pub fn read_usize(bytes: &[u8], offset: &mut usize) -> usize {
    let index_bytes: [u8; std::mem::size_of::<usize>()] = bytes[*offset..*offset + std::mem::size_of::<usize>()]
        .try_into()
        .unwrap_or_else(|e| unreachable!("Could not convert slice into array: {e:?}"));
    *offset += std::mem::size_of::<usize>();
    usize::from_le_bytes(index_bytes)
}

#[cfg(test)]
pub mod tests {
    use distances::strings::needleman_wunsch;
    use test_case::test_case;

    use crate::{
        cakes::{Algorithm, OffBall},
        Ball, Cluster, FlatVec, Metric, Partition,
    };

    use super::{
        super::search::tests::{check_search_by_distance, check_search_by_index, gen_random_data},
        Decodable, Encodable, SquishyBall,
    };

    impl Encodable for Vec<f32> {
        fn as_bytes(&self) -> Box<[u8]> {
            self.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        }

        fn encode(&self, reference: &Self) -> Box<[u8]> {
            let diffs = reference.iter().zip(self.iter()).map(|(&a, &b)| a - b).collect();
            Self::as_bytes(&diffs)
        }
    }

    impl Decodable for Vec<f32> {
        fn from_bytes(bytes: &[u8]) -> Self {
            bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|v| {
                    v.try_into()
                        .unwrap_or_else(|e| unreachable!("Could not cast to four bytes: {e:?}"))
                })
                .map(f32::from_le_bytes)
                .collect()
        }

        fn decode(reference: &Self, bytes: &[u8]) -> Self {
            let diffs = Self::from_bytes(bytes);
            reference.iter().zip(diffs.into_iter()).map(|(&a, b)| a + b).collect()
        }
    }

    impl Encodable for String {
        fn as_bytes(&self) -> Box<[u8]> {
            self.as_bytes().to_vec().into_boxed_slice()
        }

        fn encode(&self, reference: &Self) -> Box<[u8]> {
            let table =
                needleman_wunsch::compute_table::<u32>(reference, self, distances::strings::Penalties::default());
            let (aligned_x, aligned_y) = needleman_wunsch::trace_back_recursive(&table, [reference, self]);
            serialize_edits(&needleman_wunsch::unaligned_x_to_y(&aligned_x, &aligned_y))
        }
    }

    /// Serializes a vector of edit operations into a byte array.
    fn serialize_edits(edits: &[needleman_wunsch::Edit]) -> Box<[u8]> {
        let bytes = edits.iter().flat_map(edit_to_bin).collect::<Vec<_>>();
        bytes.into_boxed_slice()
    }

    /// Encodes an edit operation into a byte array.
    ///
    /// A `Del` edit is encoded as `10` followed by the index of the edit in 14 bits.
    /// A `Ins` edit is encoded as `01` followed by the index of the edit in 14 bits and the character in 8 bits.
    /// A `Sub` edit is encoded as `11` followed by the index of the edit in 14 bits and the character in 8 bits.
    ///
    /// # Arguments
    ///
    /// * `edit`: The edit operation.
    ///
    /// # Returns
    ///
    /// A byte array encoding the edit operation.
    #[allow(clippy::cast_possible_truncation)]
    fn edit_to_bin(edit: &needleman_wunsch::Edit) -> Vec<u8> {
        let mask_6 = 0b0011_1111;
        let mask_del = 0b10;
        let mask_ins = 0b01;
        let mask_sub = 0b11;
        match edit {
            // First 2 bits for the type of edit, 14 bits for the index.
            needleman_wunsch::Edit::Del(i) => {
                let mut bytes = (*i as u16).to_be_bytes().to_vec();
                bytes[0] &= mask_6;
                bytes[0] |= mask_del;
                bytes
            }
            needleman_wunsch::Edit::Ins(i, c) => {
                // First 2 bits for the type of edit, 14 bits for the index.
                let mut bytes = (*i as u16).to_be_bytes().to_vec();
                bytes[0] &= mask_6;
                bytes[0] |= mask_ins;
                // 8 bits for the character.
                bytes.push(*c as u8);
                bytes
            }
            needleman_wunsch::Edit::Sub(i, c) => {
                // First 2 bits for the type of edit, 14 bits for the index.
                let mut bytes = (*i as u16).to_be_bytes().to_vec();
                bytes[0] &= mask_6;
                bytes[0] |= mask_sub;
                // 8 bits for the character.
                bytes.push(*c as u8);
                bytes
            }
        }
    }

    impl Decodable for String {
        fn from_bytes(bytes: &[u8]) -> Self {
            Self::from_utf8(bytes.to_vec()).unwrap_or_else(|e| unreachable!("Could not cast back to string: {e:?}"))
        }

        fn decode(reference: &Self, bytes: &[u8]) -> Self {
            needleman_wunsch::apply_edits(reference, &deserialize_edits(bytes))
        }
    }

    /// Deserializes a byte array into a vector of edit operations.
    ///
    /// A `Del` edit is encoded as `10` followed by the index of the edit in 14 bits.
    /// A `Ins` edit is encoded as `01` followed by the index of the edit in 14 bits and the character in 8 bits.
    /// A `Sub` edit is encoded as `11` followed by the index of the edit in 14 bits and the character in 8 bits.
    ///
    /// # Arguments
    ///
    /// * `bytes`: The byte array encoding the edit operations.
    ///
    /// # Errors
    ///
    /// * If the byte array is not a valid encoding of edit operations.
    /// * If the edit type is not recognized.
    ///
    /// # Returns
    ///
    /// A vector of edit operations.
    fn deserialize_edits(bytes: &[u8]) -> Vec<needleman_wunsch::Edit> {
        let mut edits = Vec::new();
        let mut i = 0;
        let mask_edit = 0b11;
        while i < bytes.len() {
            let index = u16::from_be_bytes([bytes[i] & !mask_edit, bytes[i + 1]]);
            let edit_bits = bytes[i] & mask_edit;
            let edit = match edit_bits {
                0b10 => {
                    i += 2;
                    needleman_wunsch::Edit::Del(index as usize)
                }
                0b01 => {
                    let c = bytes[i + 2];
                    i += 3;
                    needleman_wunsch::Edit::Ins(index as usize, c as char)
                }
                0b11 => {
                    let c = bytes[i + 2];
                    i += 3;
                    needleman_wunsch::Edit::Sub(index as usize, c as char)
                }
                _ => unreachable!("Invalid edit type: {edit_bits:b}."),
            };
            edits.push(edit);
        }
        edits
    }

    #[test_case(1_000, 10; "1k-10")]
    #[test_case(10_000, 10; "10k-10")]
    #[test_case(100_000, 10; "100k-10")]
    #[test_case(1_000, 100; "1k-100")]
    #[test_case(10_000, 100; "10k-100")]
    fn vectors(car: usize, dim: usize) -> Result<(), String> {
        let mut algs: Vec<(
            Algorithm<f32>,
            fn(Vec<(usize, f32)>, Vec<(usize, f32)>, &str, bool) -> bool,
        )> = vec![];
        for radius in [0.1, 1.0] {
            algs.push((Algorithm::RnnClustered(radius), check_search_by_index));
        }
        for k in [1, 10, 100] {
            algs.push((Algorithm::KnnRepeatedRnn(k, 2.0), check_search_by_distance));
            algs.push((Algorithm::KnnBreadthFirst(k), check_search_by_distance));
            algs.push((Algorithm::KnnDepthFirst(k), check_search_by_distance));
        }

        let seed = 42;
        let data = gen_random_data(car, dim, 10.0, seed)?;
        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let seed = Some(seed);
        let query = &vec![0.0; dim];

        let root = Ball::new_tree(&data, &criteria, seed);

        let mut data = data;
        let root = OffBall::from_ball_tree(root, &mut data);

        let (root, data) = SquishyBall::from_root(root, data);

        for &(alg, checker) in &algs {
            let true_hits = alg.par_linear_search(&data, query);

            if car < 100_000 {
                let pred_hits = alg.search(&data, &root, query);
                checker(true_hits.clone(), pred_hits, &alg.name(), false);
            }

            let pred_hits = alg.par_search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);
        }

        Ok(())
    }

    #[test]
    fn strings() -> Result<(), String> {
        let mut algs: Vec<(
            Algorithm<u16>,
            fn(Vec<(usize, u16)>, Vec<(usize, u16)>, &str, bool) -> bool,
        )> = vec![];
        for radius in [4, 8, 16] {
            algs.push((Algorithm::RnnClustered(radius), check_search_by_index));
        }
        for k in [1, 10, 20] {
            algs.push((Algorithm::KnnRepeatedRnn(k, 2), check_search_by_distance));
            algs.push((Algorithm::KnnBreadthFirst(k), check_search_by_distance));
            algs.push((Algorithm::KnnDepthFirst(k), check_search_by_distance));
        }

        let seed_length = 100;
        let alphabet = "ACTGN".chars().collect::<Vec<_>>();
        let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
        let penalties = distances::strings::Penalties::default();
        let num_clumps = 16;
        let clump_size = 16;
        let clump_radius = 3_u16;
        let (metadata, data) = symagen::random_edits::generate_clumped_data(
            &seed_string,
            penalties,
            &alphabet,
            num_clumps,
            clump_size,
            clump_radius,
        )
        .into_iter()
        .unzip::<_, _, Vec<_>, Vec<_>>();
        let query = &seed_string;

        let distance_fn = |a: &String, b: &String| distances::strings::levenshtein::<u16>(a, b);
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(data, metric)?.with_metadata(metadata)?;

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let seed = Some(42);
        let root = Ball::new_tree(&data, &criteria, seed);

        let mut data = data;
        let root = OffBall::from_ball_tree(root, &mut data);

        let (root, data) = SquishyBall::from_root(root, data);

        for &(alg, checker) in &algs {
            let true_hits = alg.par_linear_search(&data, query);

            let pred_hits = alg.search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);

            let pred_hits = alg.par_search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);
        }

        Ok(())
    }
}
