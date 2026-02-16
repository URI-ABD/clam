//! An `AlignedSequence` is a memory-efficient representation of a sequence in an MSA.

use crate::DistanceValue;

use super::{CostMatrix, Direction, Edit, Edits};

mod aligned_range;
mod seq_iter;

use aligned_range::AlignedRange;

/// A gap in an aligned sequence.
const GAP: char = '-';

/// A memory-efficient sequence that stores gaps as the negative space between distinct ranges of characters from an original sequence.
///
/// This type is optimized for use in wide MSAs where most of the characters are gaps.
#[must_use]
#[derive(Clone, Default, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub struct AlignedSequence {
    /// The original sequence without gaps.
    original: Vec<char>,
    /// The length of the aligned sequence, including gaps.
    aligned_length: usize,
    /// The ranges representing the positions of characters from the original sequence in the aligned sequence.
    ///
    /// This is always kept sorted by the `aligned_start` of the ranges. These ranges are non-overlapping and represent the positions of the characters from the
    /// original sequence in the aligned sequence. The gaps in the aligned sequence are represented by the negative space between these ranges and the length of
    /// the aligned sequence.
    ranges: Vec<AlignedRange>,
}

impl From<Vec<char>> for AlignedSequence {
    fn from(value: Vec<char>) -> Self {
        Self::new_aligned(value.iter().copied())
    }
}

impl From<&[char]> for AlignedSequence {
    fn from(value: &[char]) -> Self {
        Self::new_aligned(value.iter().copied())
    }
}

impl From<String> for AlignedSequence {
    fn from(value: String) -> Self {
        Self::new_aligned(value.chars())
    }
}

impl From<&str> for AlignedSequence {
    fn from(value: &str) -> Self {
        Self::new_aligned(value.chars())
    }
}

impl core::fmt::Display for AlignedSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = self.iter().collect::<String>();
        write!(f, "{string}")
    }
}

impl AlignedSequence {
    /// Create a new `AlignedSequence` from the given original sequence, and have it be ready to have gaps inserted into it.
    pub fn new_unaligned<T: Into<Vec<char>>>(original: T) -> Self {
        let original = original.into();
        let length = original.len();
        Self {
            original,
            aligned_length: length,
            ranges: vec![AlignedRange::from_length(length)],
        }
    }

    /// Create a new `AlignedSequence` from the sequence of characters, accounting for any gaps that are present in the given sequence.
    pub fn new_aligned<T: Iterator<Item = char>>(aligned: T) -> Self {
        let mut original = Vec::new();
        let mut ranges = Vec::new();
        let mut current_range: Option<AlignedRange> = None;
        let mut aligned_length = 0;

        for (i, c) in aligned.enumerate() {
            aligned_length += 1;
            if c == GAP {
                if let Some(range) = current_range {
                    // We have encountered the end of a range, so we need to add the range to the list of ranges and reset the current range to None.
                    ranges.push(range);
                    current_range = None;
                } else {
                    // We have encountered a gap character, but we are not currently in a range, so we can just ignore it and move on to the next character.
                }
            } else {
                // We have encountered a non-gap character, so we need to add it to the original sequence.
                original.push(c);

                if let Some(ref mut range) = current_range {
                    // We are currently in a range, so we need to increment the length of the current range to account for the new character.
                    range.increment_length();
                } else {
                    // We have encountered the start of a new range, so we need to record the start index of the range in the original sequence.
                    current_range = Some(AlignedRange::new(i, original.len() - 1, 1));
                }
            }
        }

        // If we ended the loop while we were still in a range, we need to add the final range to the list of ranges.
        if let Some(range) = current_range {
            ranges.push(range);
        }

        Self {
            original,
            aligned_length,
            ranges,
        }
    }

    /// Returns a reference to the original sequence without gaps.
    #[must_use]
    pub fn original(&self) -> String {
        self.original.iter().filter(|&&c| c != GAP).collect()
    }

    /// Returns the length of the aligned sequence, including gaps.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.aligned_length
    }

    /// Returns true if the aligned sequence is empty (has no characters or gaps).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.aligned_length == 0
    }

    /// Returns the number of gaps in the aligned sequence.
    #[must_use]
    pub const fn gap_count(&self) -> usize {
        self.aligned_length - self.original.len()
    }

    /// Returns the number of contiguous chunks of characters from the original sequence separated by gaps in the aligned sequence.
    #[must_use]
    pub const fn chunk_count(&self) -> usize {
        self.ranges.len()
    }

    /// Returns the `Ok(i)` of the range that contains the given index in the aligned sequence or the `Err(i)` of the range that is immediately to the left of
    /// the given index if it is not contained in any range.
    fn find_range(&self, index: usize) -> Result<usize, usize> {
        match self.ranges.binary_search_by(|r| {
            if r.aligned_range().contains(&index) {
                std::cmp::Ordering::Equal
            } else if r.aligned_start() > index {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }) {
            Ok(i) => Ok(i),
            Err(i) => Err(i.saturating_sub(1)),
        }
    }

    /// Inserts a gap at the given index in the aligned sequence, shifting the characters to the right of the gap.
    fn insert_gap(&mut self, index: usize) {
        // We first need to find whether the gap is being inserted in the middle of an existing range or between two ranges.
        let i = match self.find_range(index) {
            Ok(i) => {
                // The gap is being inserted in the middle of a range, so we need to split the range into two ranges on either side of the gap.
                let right = self.ranges[i].split(index).unwrap_or_else(|err| {
                    unreachable!("The `find_range` method guarantees that the index is within the range, so the split should never fail. Got err {err}")
                });
                // We insert the new range to the right of the original range, so we can keep the `Vec` of ranges sorted by aligned_start.
                self.ranges.insert(i + 1, right);
                i
            }
            Err(i) => i, // The gap is being inserted between two ranges.
        };
        // We now need to increment the aligned_start of all ranges to the right of the inserted gap, since the gap shifts all characters to the right of it by
        // one position in the aligned sequence.
        for r in self.ranges.iter_mut().skip(i + 1) {
            r.increment_aligned_start();
        }
        // Finally, we need to increment the aligned length of the sequence to account for the newly inserted gap.
        self.aligned_length += 1;
    }

    /// Computes the dynamic programming table using the Needleman-Wunsch algorithm and the given cost matrix.
    pub fn nw_table<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> Vec<Vec<(T, Direction)>> {
        let seq1 = self.iter().collect::<Vec<_>>();
        let seq2 = other.iter().collect::<Vec<_>>();
        cost_matrix.nw_table(&seq1, &seq2)
    }

    /// Compute the Needleman-Wunsch distance between this sequence and another sequence.
    pub fn nw_distance<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> T {
        let seq1 = self.iter().collect::<Vec<_>>();
        let seq2 = other.iter().collect::<Vec<_>>();
        cost_matrix.nw_distance(&seq1, &seq2)
    }

    /// Compute the Hamming distance between this sequence and another sequence.
    pub fn hamming_distance<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> T {
        self.iter()
            .zip(other.iter())
            .filter_map(|(a, b)| if a == b { None } else { Some(cost_matrix.sub_cost(a, b)) })
            .fold(T::zero(), |acc, cost| acc + cost)
    }

    /// Compute the Needleman-Wunsch distance between the original versions of this sequence and another sequence.
    pub fn original_nw_distance<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> T {
        cost_matrix.nw_distance(&self.original, &other.original)
    }

    /// Computes the edits needed to transform the two sequences into each other based on the given DP table.
    ///
    /// # Returns
    ///
    /// A pair of `Edits`, where the first element is the edits to transform `self` into `other`, and the second element is the edits to transform `other` into
    /// `self`.
    pub fn nw_edit_scripts<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> [Edits; 2] {
        let seq1 = self.iter().collect::<Vec<_>>();
        let seq2 = other.iter().collect::<Vec<_>>();
        cost_matrix.nw_edit_scripts(&seq1, &seq2)
    }

    /// Applies the given edits to the sequence.
    pub fn apply_edits(&self, edits: &Edits) -> Self {
        let mut result = self.iter().collect::<Vec<_>>();
        result.reserve(edits.len());

        let mut offset = 0;
        for (i, edit) in edits.iter() {
            match edit {
                Edit::Sub(c) => {
                    result[i + offset] = *c;
                }
                Edit::Ins(c) => {
                    result.insert(i + offset, *c);
                    offset += 1;
                }
                Edit::Del => {
                    result.remove(i + offset);
                    offset -= 1;
                }
            }
        }

        Self::from(result)
    }

    /// Returns the indices where gaps should be inserted to align the two sequences to each other.
    #[must_use]
    pub fn compute_gap_indices<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> [Vec<usize>; 2] {
        let seq1 = self.iter().collect::<Vec<_>>();
        let seq2 = other.iter().collect::<Vec<_>>();
        cost_matrix.nw_gap_indices(&seq1, &seq2)
    }

    /// Returns a new `AlignedSequence` with gaps inserted at the specified indices in the aligned sequence.
    pub fn with_gaps(&self, indices: &[usize]) -> Self {
        let mut seq = self.clone();
        seq.insert_gaps(indices);
        seq
    }

    /// Inserts gaps at the specified indices in the sequence.
    pub fn insert_gaps(&mut self, indices: &[usize]) {
        for &idx in indices.iter().rev() {
            self.insert_gap(idx);
        }
    }
}

#[cfg(feature = "serde")]
impl databuf::Encode for AlignedSequence {
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        let original = self.original.iter().collect::<String>();
        original.encode::<CONFIG>(buffer)?;

        self.aligned_length.encode::<CONFIG>(buffer)?;

        let ranges = self.ranges.clone().into_boxed_slice();
        ranges.encode::<CONFIG>(buffer)
    }
}

#[cfg(feature = "serde")]
impl<'de> databuf::Decode<'de> for AlignedSequence {
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let original = String::decode::<CONFIG>(buffer)?.chars().collect::<Vec<_>>();
        let aligned_length = usize::decode::<CONFIG>(buffer)?;
        let ranges = Box::<[AlignedRange]>::decode::<CONFIG>(buffer)?.into_vec();
        Ok(Self {
            original,
            aligned_length,
            ranges,
        })
    }
}
