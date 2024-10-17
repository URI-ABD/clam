//! A wrapper around a `String` to use in multiple sequence alignment.

use distances::Number;

use super::Aligner;

/// A wrapper around a `String` to use in multiple sequence alignment.
#[derive(Clone)]
pub struct Sequence<'a, T: Number> {
    /// The wrapped string.
    seq: String,
    /// The aligner to use among the sequences.
    aligner: Option<&'a Aligner<T>>,
}

#[cfg(feature = "disk-io")]
impl<T: Number> bitcode::Encode for Sequence<'_, T> {
    const ENCODE_MAX: usize = String::ENCODE_MAX;
    const ENCODE_MIN: usize = String::ENCODE_MIN;

    fn encode(
        &self,
        encoding: impl bitcode::encoding::Encoding,
        writer: &mut impl bitcode::write::Write,
    ) -> bitcode::Result<()> {
        self.seq.encode(encoding, writer)
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number> bitcode::Decode for Sequence<'_, T> {
    const DECODE_MAX: usize = String::DECODE_MAX;
    const DECODE_MIN: usize = String::DECODE_MIN;

    fn decode(
        encoding: impl bitcode::encoding::Encoding,
        reader: &mut impl bitcode::read::Read,
    ) -> bitcode::Result<Self> {
        let seq = String::decode(encoding, reader)?;
        Ok(Self { seq, aligner: None })
    }
}

impl<'a, T: Number> Sequence<'a, T> {
    /// Creates a new `Sequence` from a string.
    ///
    /// # Arguments
    ///
    /// * `seq`: The string to wrap.
    /// * `aligner`: The aligner to use among the sequences.
    ///
    /// # Returns
    ///
    /// The wrapped string.
    #[must_use]
    pub const fn new(seq: String, aligner: Option<&'a Aligner<T>>) -> Self {
        Self { seq, aligner }
    }

    /// Returns the length of the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq.len()
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// Returns the aligner for the sequence.
    #[must_use]
    pub const fn aligner(&self) -> Option<&'a Aligner<T>> {
        self.aligner
    }

    /// Sets the aligner for the sequence.
    #[must_use]
    pub const fn with_aligner(mut self, aligner: &'a Aligner<T>) -> Self {
        self.aligner = Some(aligner);
        self
    }

    /// Returns the sequence
    #[must_use]
    pub fn seq(&self) -> &str {
        &self.seq
    }
}

impl<T: Number> core::fmt::Debug for Sequence<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.seq)
    }
}

impl<T: Number> PartialEq for Sequence<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}

impl<T: Number> Eq for Sequence<'_, T> {}

impl<T: Number> AsRef<str> for Sequence<'_, T> {
    fn as_ref(&self) -> &str {
        &self.seq
    }
}

impl<T: Number> AsRef<[u8]> for Sequence<'_, T> {
    fn as_ref(&self) -> &[u8] {
        self.seq.as_bytes()
    }
}
