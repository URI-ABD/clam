use crate::prelude::*;

/// A sub-trait of `Metric` that allows us to encode one instance in terms of
/// another and decode an instance from a reference and an encoding.
pub trait CodecMetric<T: Number>: Metric<T> {
    /// Encodes the target instance in terms of the reference and produces the
    /// encoding as a vec of bytes.
    fn encode(&self, reference: &[T], target: &[T]) -> Vec<u8>;

    /// Decodes a target instance from a reference instance and an encoding.
    fn decode(&self, reference: &[T], encoding: &[u8]) -> Result<Vec<T>, String>;
}

impl<T: Number> CodecMetric<T> for crate::metric::Hamming {
    fn encode(&self, x: &[T], y: &[T]) -> Vec<u8> {
        x.iter()
            .zip(y.iter())
            .enumerate()
            .filter(|(_, (&l, &r))| l != r)
            .flat_map(|(i, (_, &r))| {
                let mut i = (i as u64).to_be_bytes().to_vec();
                i.append(&mut r.to_le_bytes());
                i
            })
            .collect()
    }

    fn decode(&self, x: &[T], y: &[u8]) -> Result<Vec<T>, String> {
        let mut x = x.to_vec();
        let step = (8 + T::num_bytes()) as usize;
        if y.len() % step != 0 {
            Err("y has an incorrect number of bytes and cannot be decoded.".to_string())
        } else {
            let edits: Result<Vec<(usize, T)>, String> = y
                .chunks(step)
                .map(|chunk| {
                    let (index, value) = chunk.split_at(std::mem::size_of::<u64>());
                    let index = u64::from_be_bytes(index.try_into().unwrap()) as usize;
                    let value = T::from_le_bytes(value)?;
                    Ok((index, value))
                })
                .collect();
            edits?.into_iter().for_each(|(index, value)| {
                x[index] = value;
            });
            Ok(x)
        }
    }
}
