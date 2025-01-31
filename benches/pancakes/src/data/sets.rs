//! `MembershipSet` is a set of integers stored as a `Vec<usize>`. They use
//! set difference to encode one set in terms of another.

use std::collections::HashSet;

use abd_clam::pancakes::{Decoder, Encoder, ParDecoder, ParEncoder};
use distances::Number;

use crate::metrics::Jaccard;

/// A set of integers stored as a `Vec<usize>`.
#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct MembershipSet(Vec<usize>);

impl PartialEq for MembershipSet {
    fn eq(&self, other: &Self) -> bool {
        self.as_set() == other.as_set()
    }
}

impl MembershipSet {
    /// Returns the `MembershipSet` as a `HashSet<usize>`.
    fn as_set(&self) -> HashSet<usize> {
        self.0.iter().copied().collect()
    }
}

impl From<HashSet<usize>> for MembershipSet {
    fn from(v: HashSet<usize>) -> Self {
        Self(v.into_iter().collect())
    }
}

impl From<Vec<usize>> for MembershipSet {
    fn from(set: Vec<usize>) -> Self {
        Self(set)
    }
}

impl AsRef<[usize]> for MembershipSet {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl ParEncoder<MembershipSet> for Jaccard {}
impl ParDecoder<MembershipSet> for Jaccard {}

impl Encoder<MembershipSet> for Jaccard {
    fn to_byte_array(&self, item: &MembershipSet) -> Box<[u8]> {
        let bytes = item.0.iter().flat_map(|&x| x.to_le_bytes()).collect::<Vec<_>>();
        bytes.into_boxed_slice()
    }

    fn encode(&self, item: &MembershipSet, reference: &MembershipSet) -> Box<[u8]> {
        let s = item.as_set();
        let r = reference.as_set();

        // Find the items in `self` that are not in `reference`.
        let s_d_r = s.difference(&r).flat_map(|&x| x.to_le_bytes()).collect::<Vec<_>>();
        // Find the items in `reference` that are not in `self`.
        let r_d_s = r.difference(&s).flat_map(|&x| x.to_le_bytes()).collect::<Vec<_>>();

        // The first 8 bytes encode the length of `s_d_r`.
        // The next `s_d_r.len() * 8` bytes encode the items in `s_d_r`.
        // The remaining bytes encode the items in `r_d_s`.
        let mut bytes = Vec::with_capacity(core::mem::size_of::<usize>() + s_d_r.len() + r_d_s.len());
        bytes.extend_from_slice(&s_d_r.len().to_le_bytes());
        bytes.extend_from_slice(&s_d_r);
        bytes.extend_from_slice(&r_d_s);
        bytes.into_boxed_slice()
    }
}

impl Decoder<MembershipSet> for Jaccard {
    fn from_byte_array(&self, bytes: &[u8]) -> MembershipSet {
        MembershipSet(
            bytes
                .chunks_exact(core::mem::size_of::<usize>())
                .map(<usize as Number>::from_le_bytes)
                .collect(),
        )
    }

    fn decode(&self, bytes: &[u8], reference: &MembershipSet) -> MembershipSet {
        let n_sdr = <usize as Number>::from_le_bytes(&bytes[..core::mem::size_of::<usize>()]);
        let s_d_r = bytes[core::mem::size_of::<usize>()..(n_sdr + core::mem::size_of::<usize>())]
            .chunks_exact(core::mem::size_of::<usize>())
            .map(<usize as Number>::from_le_bytes)
            .collect::<HashSet<_>>();
        let r_d_s = bytes[(n_sdr + core::mem::size_of::<usize>())..]
            .chunks_exact(core::mem::size_of::<usize>())
            .map(<usize as Number>::from_le_bytes)
            .collect::<HashSet<_>>();

        let mut set = reference.0.clone();
        set.retain(|x| !r_d_s.contains(x));
        set.extend(s_d_r);
        MembershipSet(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membership_set() {
        let metric = Jaccard;
        let set = MembershipSet::from(vec![1, 2, 3, 4, 5]);
        let bytes = metric.to_byte_array(&set);
        let decoded = metric.from_byte_array(&bytes);
        assert_eq!(set, decoded);
    }

    #[test]
    fn test_membership_set_encode() {
        let metric = Jaccard;
        let set = MembershipSet::from(vec![2, 3, 4, 5]);
        let reference = MembershipSet::from(vec![1, 2, 3]);
        let bytes = metric.encode(&set, &reference);
        let decoded = metric.decode(&bytes, &reference);
        assert_eq!(set, decoded);
    }
}
