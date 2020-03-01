use rayon::prelude::*;

pub fn hamming<T: PartialEq + Sync + Send>(x: &[T], y: &[T]) -> u64 {
    x.par_iter()
        .zip(y.par_iter())
        .map(|(a, b)| if a == b { 0 } else { 1 })
        .sum()
}
