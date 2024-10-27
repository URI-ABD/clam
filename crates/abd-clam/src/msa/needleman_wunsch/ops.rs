//! Alignment operations for Needleman-Wunsch algorithm.

/// The direction of the edit operation in the DP table.
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Direction {
    /// Diagonal (Up and Left) for a match or substitution.
    Diagonal,
    /// Up for a gap in the first sequence.
    Up,
    /// Left for a gap in the second sequence.
    Left,
}

/// The type of edit operation.
pub enum Edit {
    /// Substitution of one character for another.
    Sub(u8),
    /// Insertion of a character.
    Ins(u8),
    /// Deletion of a character.
    Del,
}

/// The sequence of edits needed to turn one unaligned sequence into another.
pub struct Edits {
    /// The edits and the corresponding positions in the sequences.
    edits: Vec<(usize, Edit)>,
}

impl IntoIterator for Edits {
    type Item = (usize, Edit);
    type IntoIter = std::vec::IntoIter<(usize, Edit)>;

    fn into_iter(self) -> Self::IntoIter {
        self.edits.into_iter()
    }
}

impl FromIterator<(usize, Edit)> for Edits {
    fn from_iter<I: IntoIterator<Item = (usize, Edit)>>(iter: I) -> Self {
        Self {
            edits: iter.into_iter().collect(),
        }
    }
}
