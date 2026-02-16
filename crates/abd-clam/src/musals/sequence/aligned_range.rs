//! A helper struct to represent a range of characters from the original sequence in the aligned sequence.

/// A helper struct to represent a range of characters from the original sequence in the aligned sequence.
///
/// These are ordered by the `aligned_start` of the range, and are always kept in sorted order in the `AlignedSequence`.
#[must_use]
#[derive(Clone, Default, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub struct AlignedRange {
    /// The starting index of the range in the aligned sequence.
    aligned_start: usize,
    /// The starting index of the range in the original sequence.
    original_start: usize,
    /// The length of the range (number of characters from the original sequence).
    length: usize,
}

impl PartialOrd for AlignedRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AlignedRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.aligned_start.cmp(&other.aligned_start)
    }
}

impl AlignedRange {
    /// Creates a new `AlignedRange` that represents the entire original sequence in the aligned sequence.
    pub const fn from_length(length: usize) -> Self {
        Self {
            aligned_start: 0,
            original_start: 0,
            length,
        }
    }

    /// Creates a new `AlignedRange` with the given starting indices and length.
    pub const fn new(aligned_start: usize, original_start: usize, length: usize) -> Self {
        Self {
            aligned_start,
            original_start,
            length,
        }
    }

    /// Returns the starting index of the range in the aligned sequence.
    #[must_use]
    pub const fn aligned_start(&self) -> usize {
        self.aligned_start
    }

    /// Returns the starting index of the range in the original sequence.
    #[must_use]
    pub const fn original_start(&self) -> usize {
        self.original_start
    }

    /// Gets the range of indices in the original sequence that this range corresponds to.
    #[must_use]
    pub const fn original_range(&self) -> std::ops::Range<usize> {
        self.original_start..(self.original_start + self.length)
    }

    /// Increments the starting indices of the range by one, effectively shifting the range to the right in the aligned sequence.
    ///
    /// This is used when inserting a gap before the range, which shifts the positions of the characters in the aligned sequence.
    pub const fn increment_aligned_start(&mut self) {
        self.aligned_start += 1;
    }

    /// Increments the length of the range by one, effectively extending the range to include one more character from the original sequence.
    pub const fn increment_length(&mut self) {
        self.length += 1;
    }

    /// Returns the end index of the range in the aligned sequence.
    #[must_use]
    pub const fn aligned_end(&self) -> usize {
        self.aligned_start + self.length
    }

    /// Returns the range of indices in the aligned sequence that this range corresponds to.
    #[must_use]
    pub const fn aligned_range(&self) -> std::ops::Range<usize> {
        self.aligned_start..self.aligned_end()
    }

    /// Splits the range at the given index, creating a new range for the second part.
    ///
    /// This is used when inserting a gap in the middle of the range, which splits the characters into two separate ranges in the aligned sequence.
    ///
    /// # Errors
    ///
    /// - If the index is out of bounds (not within the range), an error is returned.
    pub fn split(&mut self, index: usize) -> Result<Self, String> {
        if self.aligned_range().contains(&index) {
            let left_range = Self {
                aligned_start: self.aligned_start,
                original_start: self.original_start,
                length: index - self.aligned_start,
            };

            let right_range = Self {
                aligned_start: index,
                original_start: self.original_start + left_range.length,
                length: self.length - left_range.length,
            };

            *self = left_range;
            Ok(right_range)
        } else {
            Err(format!("Index {index} is out of bounds for range {:?}", self.aligned_range()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AlignedRange;

    #[test]
    fn aligned_range() -> Result<(), String> {
        let mut range = AlignedRange::from_length(5);
        assert_eq!(range.aligned_end(), 5);

        range.increment_aligned_start();
        assert_eq!(range.aligned_start, 1);
        assert_eq!(range.original_start, 0);
        assert_eq!(range.aligned_end(), 6);

        for i in [0, 6] {
            let split = range.clone().split(i);
            assert!(split.is_err(), "Expected error for index {i}, but got: {split:?}");
        }
        for i in 1..6 {
            let split = range.clone().split(i);
            assert!(split.is_ok(), "Failed to split at index {i}: error: {split:?}");
        }

        let right = range.split(4)?;

        assert_eq!(range.aligned_start, 1);
        assert_eq!(range.original_start, 0);
        assert_eq!(range.length, 3);
        assert_eq!(range.aligned_end(), 4);

        assert_eq!(right.aligned_start, 4);
        assert_eq!(right.original_start, 3);
        assert_eq!(right.length, 2);
        assert_eq!(right.aligned_end(), 6);

        Ok(())
    }
}
