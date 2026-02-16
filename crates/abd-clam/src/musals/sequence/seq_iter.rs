//! An iterator over the characters of an `AlignedSequence`, including gaps.

use core::iter::FusedIterator;

use super::{AlignedSequence, GAP};

/// An iterator over the characters of an `AlignedSequence`, including gaps.
pub struct AlignedSequenceIter<'a> {
    /// An iterator over the characters of the aligned sequence, including gaps.
    chars_iter: Box<dyn Iterator<Item = char> + 'a>,
    /// The number of remaining characters in the aligned sequence, including gaps. This exists to allow us to implement `size_hint` for the iterator.
    remaining: usize,
}

impl Iterator for AlignedSequenceIter<'_> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.chars_iter.next();
        if result.is_some() {
            self.remaining -= 1;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> IntoIterator for &'a AlignedSequence {
    type Item = char;
    type IntoIter = AlignedSequenceIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FusedIterator for AlignedSequenceIter<'_> {}

impl AlignedSequence {
    /// Returns an iterator over the characters of the aligned sequence, including gaps.
    #[must_use]
    pub fn iter(&self) -> AlignedSequenceIter<'_> {
        if self.ranges.len() == 1 {
            // The sequence has no gaps in the middle, so we need only consider whether there are gaps at the start or end of the sequence.

            if self.aligned_length == self.original.len() {
                // No gaps, so we can just return an iterator over the original sequence.
                return AlignedSequenceIter {
                    chars_iter: Box::new(self.original.iter().copied()),
                    remaining: self.aligned_length,
                };
            }

            // The sequence has gaps at the start or end (or both).
            let (start, end) = (self.ranges[0].aligned_start(), self.ranges[0].aligned_end());
            let start = core::iter::repeat_n(GAP, start);
            let middle = self.original.iter().copied();
            let end = core::iter::repeat_n(GAP, self.aligned_length - end);
            return AlignedSequenceIter {
                chars_iter: Box::new(start.chain(middle).chain(end)),
                remaining: self.aligned_length,
            };
        }

        // The sequence has gaps in the middle, so we need to consider the gaps between the ranges as well.

        let first_range = self.ranges.first().unwrap_or_else(|| unreachable!("We checked len == 1 above."));
        let start = core::iter::repeat_n(GAP, first_range.aligned_start());

        let middle = self.ranges.array_windows::<2>().flat_map(|[l, r]| {
            let delta = r.aligned_start() - l.aligned_end();
            let gaps = core::iter::repeat_n(GAP, delta);
            let chars = self.original[l.original_range()].iter().copied();
            chars.chain(gaps)
        });

        let last_range = self.ranges.last().unwrap_or_else(|| unreachable!("We checked len == 1 above."));
        let last = self.original[last_range.original_start()..].iter().copied();
        let end = core::iter::repeat_n(GAP, self.aligned_length - last_range.aligned_range().end);

        AlignedSequenceIter {
            chars_iter: Box::new(start.chain(middle).chain(last).chain(end)),
            remaining: self.aligned_length,
        }
    }
}
