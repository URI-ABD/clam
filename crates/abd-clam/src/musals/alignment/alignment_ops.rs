//! Alignment operations for Needleman-Wunsch algorithm.

/// The direction of the edit operation in the DP table.
#[must_use]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub enum Direction {
    /// Diagonal (Up and Left) for a match or substitution.
    Diagonal,
    /// Up for a gap in the first sequence.
    Up,
    /// Left for a gap in the second sequence.
    Left,
}

/// The type of edit operation.
#[must_use]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub enum Edit {
    /// Substitution of one character for another.
    Sub(char),
    /// Insertion of a character.
    Ins(char),
    /// Deletion of a character.
    Del,
}

/// The sequence of edits needed to turn one unaligned sequence into another.
///
/// The `Edits` are a vector of tuples, where each tuple contains the index at which the edit occurs *in the original sequence*, and the `Edit` operation to be
/// applied at that index.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub struct Edits(pub Vec<(usize, Edit)>);

impl core::ops::Deref for Edits {
    type Target = [(usize, Edit)];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Edits {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "serde")]
impl databuf::Encode for Edits {
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        let edits = self.0.clone().into_boxed_slice();
        edits.encode::<CONFIG>(buffer)
    }
}

#[cfg(feature = "serde")]
impl<'de> databuf::Decode<'de> for Edits {
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let edits = Box::<[(usize, Edit)]>::decode::<CONFIG>(buffer)?;
        Ok(Self(edits.into_vec()))
    }
}

/// The sequence of substitutions needed to turn one aligned sequence into aligned sequence.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub struct Substitutions(pub Vec<(usize, char)>);

impl core::ops::Deref for Substitutions {
    type Target = [(usize, char)];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Substitutions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "serde")]
impl databuf::Encode for Substitutions {
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        let (indices, chars): (Vec<_>, String) = self.0.iter().copied().unzip();

        let indices_boxed = indices.into_boxed_slice();
        indices_boxed.encode::<CONFIG>(buffer)?;

        let chars_boxed = chars.into_boxed_str();
        chars_boxed.encode::<CONFIG>(buffer)
    }
}

#[cfg(feature = "serde")]
impl<'de> databuf::Decode<'de> for Substitutions {
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let indices = Box::<[usize]>::decode::<CONFIG>(buffer)?;
        let chars = Box::<str>::decode::<CONFIG>(buffer)?;
        if indices.len() == chars.len() {
            let substitutions = indices.into_iter().zip(chars.chars()).collect();
            Ok(Self(substitutions))
        } else {
            Err(format!("Mismatched lengths for indices and characters: {} vs {}", indices.len(), chars.len()).into())
        }
    }
}
