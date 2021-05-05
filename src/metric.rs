//! Number trait and Metric type.

use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::sync::Arc;

use ndarray::{ArrayView, IxDyn};
use ndarray_npy::{ReadableElement, WritableElement};
use num_traits::{Num, NumCast};

/// Use the Number trait to compute distances among generic numeric types.
pub trait Number:
    Num + NumCast + Sum + Copy + Clone + PartialOrd + Send + Sync + Debug + Display + ReadableElement + WritableElement
{
}
impl Number for f32 {}
impl Number for f64 {}
impl Number for u8 {}
impl Number for u16 {}
impl Number for u32 {}
impl Number for u64 {}
impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}

/// A Metric is a function that takes two instances (generic over a Number T)
/// and produces a single non-negative Number U.
pub type Metric<T, U> = fn(Arc<ArrayView<T, IxDyn>>, Arc<ArrayView<T, IxDyn>>) -> U;

/// Returns a Metric from a given name, or an Err if the name is not found among the implemented metrics.
///
/// # Arguments
///
/// * metric: A `&str` name of a distance function.
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
pub fn metric_new<T: Number, U: Number>(metric: &'static str) -> Result<Metric<T, U>, String> {
    match metric {
        "euclidean" => Ok(euclidean),
        "euclideansq" => Ok(euclideansq),
        "manhattan" => Ok(manhattan),
        "cosine" => Ok(cosine),
        "hamming" => Ok(hamming),
        "jaccard" => Ok(jaccard),
        _ => Err(format!("{} is not defined as a metric.", metric)),
    }
}

fn euclidean<T: Number, U: Number>(x: Arc<ArrayView<T, IxDyn>>, y: Arc<ArrayView<T, IxDyn>>) -> U {
    let d: f64 = NumCast::from(euclideansq::<T, U>(x, y)).unwrap();
    U::from(d.sqrt()).unwrap()
}

fn euclideansq<T: Number, U: Number>(x: Arc<ArrayView<T, IxDyn>>, y: Arc<ArrayView<T, IxDyn>>) -> U {
    let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
    U::from(d).unwrap()
}

fn manhattan<T: Number, U: Number>(x: Arc<ArrayView<T, IxDyn>>, y: Arc<ArrayView<T, IxDyn>>) -> U {
    let d: T = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| if a > b { a - b } else { b - a })
        .sum();
    U::from(d).unwrap()
}

fn dot<T: Number>(x: Arc<ArrayView<T, IxDyn>>, y: Arc<ArrayView<T, IxDyn>>) -> T {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

#[allow(clippy::suspicious_operation_groupings)]
fn cosine<T: Number, U: Number>(x: Arc<ArrayView<T, IxDyn>>, y: Arc<ArrayView<T, IxDyn>>) -> U {
    let xx = dot(Arc::clone(&x), Arc::clone(&x));
    if xx == T::zero() {
        return U::one();
    }

    let yy = dot(Arc::clone(&y), Arc::clone(&y));
    if yy == T::zero() {
        return U::one();
    }

    let xy = dot(Arc::clone(&x), Arc::clone(&y));
    if xy <= T::zero() {
        return U::one();
    }

    let similarity: f64 = NumCast::from(xy * xy / (xx * yy)).unwrap();
    U::one() - U::from(similarity.sqrt()).unwrap()
}

fn hamming<T: Number, U: Number>(x: Arc<ArrayView<T, IxDyn>>, y: Arc<ArrayView<T, IxDyn>>) -> U {
    let d = x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count();
    U::from(d).unwrap()
}

fn jaccard<T: Number, U: Number>(x: Arc<ArrayView<T, IxDyn>>, y: Arc<ArrayView<T, IxDyn>>) -> U {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use float_cmp::approx_eq;
    use ndarray::{arr2, Array2};

    use crate::prelude::*;

    #[test]
    fn test_on_real() {
        let data: Array2<f64> = arr2(&[[1., 2., 3.], [3., 3., 1.]]);

        let distance = metric_new("euclideansq").unwrap();
        approx_eq!(
            f64,
            distance(Arc::new(data.row(0).into_dyn()), Arc::new(data.row(0).into_dyn())),
            0.
        );
        approx_eq!(
            f64,
            distance(Arc::new(data.row(0).into_dyn()), Arc::new(data.row(1).into_dyn())),
            9.
        );

        let distance = metric_new("euclidean").unwrap();
        approx_eq!(
            f64,
            distance(Arc::new(data.row(0).into_dyn()), Arc::new(data.row(0).into_dyn())),
            0.
        );
        approx_eq!(
            f64,
            distance(Arc::new(data.row(0).into_dyn()), Arc::new(data.row(1).into_dyn())),
            3.
        );

        let distance = metric_new("manhattan").unwrap();
        approx_eq!(
            f64,
            distance(Arc::new(data.row(0).into_dyn()), Arc::new(data.row(0).into_dyn())),
            0.
        );
        approx_eq!(
            f64,
            distance(Arc::new(data.row(0).into_dyn()), Arc::new(data.row(1).into_dyn())),
            5.
        );
    }

    #[test]
    fn test_panic() {
        let f = metric_new::<f32, f32>("aloha");
        match f {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        };
    }
}
