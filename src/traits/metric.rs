//! A `Metric` allows for calculating distances between instances in a `Dataset`.

use std::collections::HashSet;
use std::convert::TryInto;

use num_traits::NumCast;
use rayon::prelude::*;

use crate::Number;

/// A `Metric` is a function that takes two instances (generic over a `Number` T)
/// from a `Dataset` and deterministically produces a non-negative `Number` U.
///
/// Optionally, a `Metric` also allows us to encode one instance in terms of another
/// and decode an instance from a reference and an encoding.
pub trait Metric<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    /// Returns the name of the `Metric` as a String.
    fn name(&self) -> String;

    /// Returns the distance between two instances.
    fn one_to_one(&self, x: &[T], y: &[T]) -> U;

    fn one_to_many(&self, x: &[T], ys: &[Vec<T>]) -> Vec<U> {
        ys.iter().map(|y| self.one_to_one(x, y)).collect()
    }

    fn par_one_to_many(&self, x: &[T], ys: &[Vec<T>]) -> Vec<U> {
        ys.par_iter().map(|y| self.one_to_one(x, y)).collect()
    }

    fn many_to_many(&self, xs: &[Vec<T>], ys: &[Vec<T>]) -> Vec<Vec<U>> {
        xs.iter().map(|x| self.one_to_many(x, ys)).collect()
    }

    fn par_many_to_many(&self, xs: &[Vec<T>], ys: &[Vec<T>]) -> Vec<Vec<U>> {
        xs.par_iter().map(|x| self.one_to_many(x, ys)).collect()
    }

    fn pairwise(&self, is: &[Vec<T>]) -> Vec<Vec<U>> {
        self.many_to_many(is, is)
    }

    fn par_pairwise(&self, is: &[Vec<T>]) -> Vec<Vec<U>> {
        self.par_many_to_many(is, is)
    }

    fn is_expensive(&self) -> bool {
        false
    }

    /// Encodes the target instance in terms of the reference and produces a vec of bytes.
    ///
    /// This method is optional and so the default just returns an Err.
    #[allow(unused_variables)]
    fn encode(&self, reference: &[T], target: &[T]) -> Result<Vec<u8>, String> {
        Err(format!("encode is not implemented for {:?}", self.name()))
    }

    /// Decodes a target instances from a reference instance and a bytes encoding.
    ///
    /// This method is optional and so the default just returns an Err.
    #[allow(unused_variables)]
    fn decode(&self, reference: &[T], encoding: &[u8]) -> Result<Vec<T>, String> {
        Err(format!("decode is not implemented for {:?}", self.name()))
    }
}

/// Returns a `Metric` from a given name, or an Err if the name
/// is not found among the implemented `Metrics`.
///
/// # Arguments
///
/// * `metric`: A `&str` name of a distance function.
/// This can be one of:
///   - "euclidean": L2-norm.
///   - "euclideansq": Squared L2-norm.
///   - "manhattan": L1-norm.
///   - "cosine": Cosine distance.
///   - "hamming": Hamming distance.
///   - "jaccard": Jaccard distance.
///
/// We plan on adding the following:
///   - "levenshtein": Edit-distance among strings (e.g. genomic/amino-acid sequences).
///   - "wasserstein": Earth-Mover-Distance among high-dimensional probability distributions (will be usable with images)
///   - "tanamoto": Jaccard distance between the Maximal-Common-Subgraph of two molecular structures.
pub fn metric_from_name<T: Number, U: Number>(metric: &str) -> Result<&dyn Metric<T, U>, String> {
    match metric {
        "euclidean" => Ok(&Euclidean),
        "euclideansq" => Ok(&EuclideanSq),
        "manhattan" => Ok(&Manhattan),
        "cosine" => Ok(&Cosine),
        "hamming" => Ok(&Hamming),
        "jaccard" => Ok(&Jaccard),
        _ => Err(format!("{} is not defined as a metric.", metric)),
    }
}

/// Implements Euclidean distance, the L2-norm.
#[derive(Debug)]
pub struct Euclidean;

impl<T: Number, U: Number> Metric<T, U> for Euclidean {
    fn name(&self) -> String {
        "euclidean".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
        let d: f64 = NumCast::from(d).unwrap();
        U::from(d.sqrt()).unwrap()
    }
}

/// Implements Squared-Euclidean distance, the squared L2-norm.
#[derive(Debug)]
pub struct EuclideanSq;

impl<T: Number, U: Number> Metric<T, U> for EuclideanSq {
    fn name(&self) -> String {
        "euclideansq".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
        U::from(d).unwrap()
    }
}

/// Implements Manhattan/Cityblock distance, the L1-norm.
#[derive(Debug)]
pub struct Manhattan;

impl<T: Number, U: Number> Metric<T, U> for Manhattan {
    fn name(&self) -> String {
        "manhattan".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d: T = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| if a > b { a - b } else { b - a })
            .sum();
        U::from(d).unwrap()
    }
}

/// Implements Cosine distance, 1 - cosine-similarity.
#[derive(Debug)]
pub struct Cosine;

fn dot<T: Number>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

impl<T: Number, U: Number> Metric<T, U> for Cosine {
    fn name(&self) -> String {
        "cosine".to_string()
    }

    #[allow(clippy::suspicious_operation_groupings)]
    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let xx = dot(x, x);
        if xx == T::zero() {
            return U::one();
        }

        let yy = dot(y, y);
        if yy == T::zero() {
            return U::one();
        }

        let xy = dot(x, y);
        if xy <= T::zero() {
            return U::one();
        }

        let similarity: f64 = NumCast::from(xy * xy / (xx * yy)).unwrap();
        U::one() - U::from(similarity.sqrt()).unwrap()
    }
}

/// Implements Hamming distance.
/// This is not normalized by the number of features.
#[derive(Debug)]
pub struct Hamming;

impl<T: Number, U: Number> Metric<T, U> for Hamming {
    fn name(&self) -> String {
        "hamming".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d = x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count();
        U::from(d).unwrap()
    }

    fn encode(&self, x: &[T], y: &[T]) -> Result<Vec<u8>, String> {
        let encoding = x
            .iter()
            .zip(y.iter())
            .enumerate()
            .filter(|(_, (&l, &r))| l != r)
            .flat_map(|(i, (_, &r))| {
                let mut i = (i as u64).to_be_bytes().to_vec();
                i.append(&mut r.to_bytes());
                i
            })
            .collect();
        Ok(encoding)
    }

    fn decode(&self, x: &[T], y: &[u8]) -> Result<Vec<T>, String> {
        let mut x = x.to_owned();
        let step = (8 + T::num_bytes()) as usize;
        y.chunks(step).for_each(|chunk| {
            let (index, value) = chunk.split_at(std::mem::size_of::<u64>());
            let index = u64::from_be_bytes(index.try_into().unwrap()) as usize;
            x[index] = T::from_bytes(value);
        });
        Ok(x)
    }
}

/// Implements Cosine distance, 1 - jaccard-similarity.
///
/// Warning: DO NOT use this with floating-point numbers.
#[derive(Debug)]
pub struct Jaccard;

impl<T: Number, U: Number> Metric<T, U> for Jaccard {
    fn name(&self) -> String {
        "jaccard".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        if x.is_empty() || y.is_empty() {
            return U::one();
        }

        let x: HashSet<u64> = x.iter().map(|&a| NumCast::from(a).unwrap()).collect();
        let intersect = y.iter().filter(|&&b| x.contains(&NumCast::from(b).unwrap())).count();

        if intersect == x.len() && intersect == y.len() {
            return U::zero();
        }

        U::one() - U::from(intersect).unwrap() / U::from(x.len() + y.len() - intersect).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use crate::metric::metric_from_name;

    #[test]
    fn test_on_real() {
        let a = vec![1., 2., 3.];
        let b = vec![3., 3., 1.];

        let metric = metric_from_name("euclideansq").unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 9.);

        let metric = metric_from_name("euclidean").unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 3.);

        let metric = metric_from_name("manhattan").unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 5.);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        let _ = metric_from_name::<f32, f32>("aloha").unwrap();
    }
}
