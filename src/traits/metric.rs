//! A `Metric` allows for calculating distances between instances in a `Dataset`.

use std::collections::HashSet;
use std::convert::TryInto;
use std::sync::Arc;

use num_traits::NumCast;

use crate::Number;

/// A `Metric` is a function that takes two instances (generic over a `Number` T)
/// from a `Dataset` and deterministically produces a non-negative `Number` U.
///
/// Optionally, a `Metric` also allows us to encode one instance in terms of another
/// and decode decode an instance froma  reference and an encoding.
pub trait Metric<T, U>: Send + Sync {
    /// Returns the name of the `Metric` as a String.
    fn name(&self) -> String;

    /// Returns the distance between two instances.
    fn distance(&self, x: &[T], y: &[T]) -> U;

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
pub fn metric_new<T: Number, U: Number>(metric: &str) -> Result<Arc<dyn Metric<T, U>>, String> {
    match metric {
        "euclidean" => Ok(Arc::new(Euclidean)),
        "euclideansq" => Ok(Arc::new(EuclideanSq)),
        "manhattan" => Ok(Arc::new(Manhattan)),
        "cosine" => Ok(Arc::new(Cosine)),
        "hamming" => Ok(Arc::new(Hamming)),
        "jaccard" => Ok(Arc::new(Jaccard)),
        _ => Err(format!("{} is not defined as a metric.", metric)),
    }
}

/// Implements Euclidean distance, the L2-norm.
pub struct Euclidean;

impl<T: Number, U: Number> Metric<T, U> for Euclidean {
    fn name(&self) -> String {
        "euclidean".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
        let d: f64 = NumCast::from(d).unwrap();
        U::from(d.sqrt()).unwrap()
    }
}

/// Implements Squared-Euclidean distance, the squared L2-norm.
pub struct EuclideanSq;

impl<T: Number, U: Number> Metric<T, U> for EuclideanSq {
    fn name(&self) -> String {
        "euclideansq".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
        U::from(d).unwrap()
    }
}

/// Implements Manhattan/Cityblock distance, the L1-norm.
pub struct Manhattan;

impl<T: Number, U: Number> Metric<T, U> for Manhattan {
    fn name(&self) -> String {
        "manhattan".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| if a > b { a - b } else { b - a }).sum();
        U::from(d).unwrap()
    }
}

/// Implements Cosine distance, 1 - cosine-similarity.
pub struct Cosine;

fn dot<T: Number>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

impl<T: Number, U: Number> Metric<T, U> for Cosine {
    fn name(&self) -> String {
        "cosine".to_string()
    }

    #[allow(clippy::suspicious_operation_groupings)]
    fn distance(&self, x: &[T], y: &[T]) -> U {
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
pub struct Hamming;

impl<T: Number, U: Number> Metric<T, U> for Hamming {
    fn name(&self) -> String {
        "hamming".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d = x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count();
        U::from(d).unwrap()
    }

    fn encode(&self, x: &[T], y: &[T]) -> Result<Vec<u8>, String> {
        let encoding = x
            .iter()
            .zip(y.iter())
            .enumerate()
            .filter(|(_, (&l, &r))| l != r)
            .map(|(i, (_, &r))| {
                let mut i = (i as u64).to_be_bytes().to_vec();
                i.append(&mut r.to_bytes());
                i
            })
            .flatten()
            .collect();
        Ok(encoding)
    }

    fn decode(&self, x: &[T], y: &[u8]) -> Result<Vec<T>, String> {
        let mut x = x.to_owned();
        let step = (8 + T::num_bytes()) as usize;
        y.chunks(step).for_each(|chunk| {
            let (index, value) = chunk.split_at(std::mem::size_of::<u64>());
            let index = u64::from_be_bytes(index.try_into().unwrap()) as usize;
            x[index] = T::from_bytes(&value.to_vec());
        });
        Ok(x)
    }
}

/// Implements Cosine distance, 1 - jaccard-similarity.
///
/// Warning: DO NOT use this with floating-point numbers.
pub struct Jaccard;

impl<T: Number, U: Number> Metric<T, U> for Jaccard {
    fn name(&self) -> String {
        "jaccard".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
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
    use ndarray::{arr2, Array2};

    use crate::metric::metric_new;

    #[test]
    fn test_on_real() {
        let data: Array2<f64> = arr2(&[[1., 2., 3.], [3., 3., 1.]]);
        let row0 = data.row(0).to_vec();
        let row1 = data.row(1).to_vec();

        let metric = metric_new("euclideansq").unwrap();
        approx_eq!(f64, metric.distance(&row0, &row0), 0.);
        approx_eq!(f64, metric.distance(&row0, &row1), 9.);

        let metric = metric_new("euclidean").unwrap();
        approx_eq!(f64, metric.distance(&row0, &row0), 0.);
        approx_eq!(f64, metric.distance(&row0, &row1), 3.);

        let metric = metric_new("manhattan").unwrap();
        approx_eq!(f64, metric.distance(&row0, &row0), 0.);
        approx_eq!(f64, metric.distance(&row0, &row1), 5.);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        let _ = metric_new::<f32, f32>("aloha").unwrap();
    }
}
