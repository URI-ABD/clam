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

impl core::fmt::Debug for Edit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sub(c) => f.debug_tuple("Sub").field(&(*c as char)).finish(),
            Self::Ins(c) => f.debug_tuple("Ins").field(&(*c as char)).finish(),
            Self::Del => write!(f, "Del"),
        }
    }
}

/// The sequence of edits needed to turn one unaligned sequence into another.
#[derive(Debug)]
pub struct Edits(Vec<(usize, Edit)>);

impl From<Vec<(usize, Edit)>> for Edits {
    fn from(edits: Vec<(usize, Edit)>) -> Self {
        Self(edits)
    }
}

impl AsRef<[(usize, Edit)]> for Edits {
    fn as_ref(&self) -> &[(usize, Edit)] {
        &self.0
    }
}

impl IntoIterator for Edits {
    type Item = (usize, Edit);
    type IntoIter = std::vec::IntoIter<(usize, Edit)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<(usize, Edit)> for Edits {
    fn from_iter<I: IntoIterator<Item = (usize, Edit)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}
