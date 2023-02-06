//! Provides the `Metric` trait and implementations for some common distance
//! functions.

use num_traits::NumCast;
// use rayon::prelude::*;

use crate::Number;

/// A `Metric` is a function that takes two instances (over a `Number` T) from a
/// `Dataset` and deterministically produces a non-negative `Number` U.
pub trait Metric<T: Number>: std::fmt::Debug + Send + Sync {
    /// Returns the name of the `Metric` as a String.
    fn name(&self) -> String;

    /// Returns the distance between two instances.
    fn one_to_one(&self, x: &[T], y: &[T]) -> f64;

    fn one_to_many(&self, x: &[T], ys: &[&[T]]) -> Vec<f64> {
        ys.iter().map(|y| self.one_to_one(x, y)).collect()
    }

    // fn par_one_to_many(&self, x: &[T], ys: &[&[T]]) -> Vec<f64> {
    //     ys.par_iter().map(|y| self.one_to_one(x, y)).collect()
    // }

    fn many_to_many(&self, xs: &[&[T]], ys: &[&[T]]) -> Vec<Vec<f64>> {
        xs.iter().map(|x| self.one_to_many(x, ys)).collect()
    }

    // fn par_many_to_many(&self, xs: &[&[T]], ys: &[&[T]]) -> Vec<Vec<f64>> {
    //     xs.par_iter().map(|x| self.one_to_many(x, ys)).collect()
    // }

    fn pairwise(&self, is: &[&[T]]) -> Vec<Vec<f64>> {
        self.many_to_many(is, is)
    }

    // fn par_pairwise(&self, is: &[&[T]]) -> Vec<Vec<f64>> {
    //     self.par_many_to_many(is, is)
    // }

    /// Whether the metric is expensive to compute.
    fn is_expensive(&self) -> bool;
}

/// Returns a `Metric` from a given name, or an Err if the name is not found
/// among the implemented `Metrics`.
///
/// # Arguments
///
/// * `name`: of the distance function.
/// This can be one of:
///     - "euclidean": L2-norm.
///     - "euclideansq": Squared L2-norm.
///     - "manhattan": L1-norm.
///     - "cosine": Cosine distance.
///     - "hamming": Hamming distance.
///     - "jaccard": Jaccard distance.
///
/// We plan on adding the following:
/// - "levenshtein": Minimum edit distance among strings (e.g.
/// genomic/amino-acid sequences).
/// - "wasserstein": Earth-Mover-Distance among high-dimensional probability
/// distributions (will be usable with images)
/// - "tanamoto": Jaccard distance between the Maximal-Common-Subgraph of two
/// molecular structures.
pub fn metric_from_name<T: Number>(name: &str, is_expensive: bool) -> Result<Box<dyn Metric<T>>, String> {
    match name {
        "euclidean" => Ok(Box::new(Euclidean { is_expensive })),
        "euclideansq" => Ok(Box::new(EuclideanSq { is_expensive })),
        "manhattan" => Ok(Box::new(Manhattan { is_expensive })),
        "cosine" => Ok(Box::new(Cosine { is_expensive })),
        "hamming" => Ok(Box::new(Hamming { is_expensive })),
        "jaccard" => Ok(Box::new(Jaccard { is_expensive })),
        _ => Err(format!("{name} is not defined as a metric.")),
    }
}

/// L2-norm.
#[derive(Debug)]
pub struct Euclidean {
    pub is_expensive: bool,
}

impl<T: Number> Metric<T> for Euclidean {
    fn name(&self) -> String {
        "euclidean".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b))
            .map(|v| (v * v).as_f64())
            .sum::<f64>()
            .sqrt()
    }

    fn is_expensive(&self) -> bool {
        self.is_expensive
    }
}

/// Squared L2-norm.
#[derive(Debug)]
pub struct EuclideanSq {
    pub is_expensive: bool,
}

impl<T: Number> Metric<T> for EuclideanSq {
    fn name(&self) -> String {
        "euclideansq".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b))
            .map(|v| (v * v).as_f64())
            .sum()
    }

    fn is_expensive(&self) -> bool {
        self.is_expensive
    }
}

/// L1-norm.
#[derive(Debug)]
pub struct Manhattan {
    pub is_expensive: bool,
}

impl<T: Number> Metric<T> for Manhattan {
    fn name(&self) -> String {
        "manhattan".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| if a > b { a - b } else { b - a })
            .map(|v| v.as_f64())
            .sum()
    }

    fn is_expensive(&self) -> bool {
        self.is_expensive
    }
}

/// 1 - cosine-similarity.
#[derive(Debug)]
pub struct Cosine {
    pub is_expensive: bool,
}

impl<T: Number> Metric<T> for Cosine {
    fn name(&self) -> String {
        "cosine".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> f64 {
        let (xx, yy, xy) = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a.as_f64(), b.as_f64()))
            .fold((0., 0., 0.), |(xx, yy, xy), (a, b)| {
                (xx + a * a, yy + b * b, xy + a * b)
            });

        if xx == 0. || yy == 0. || xy <= 0. {
            1.
        } else {
            // TODO: Try using inverse-reciprocal-sqrt method
            1. - xy / (xx * yy).sqrt()
        }
    }

    fn is_expensive(&self) -> bool {
        self.is_expensive
    }
}

/// Count of differences at each indexed feature. This is not normalized by the
/// number of features.
#[derive(Debug)]
pub struct Hamming {
    pub is_expensive: bool,
}

impl<T: Number> Metric<T> for Hamming {
    fn name(&self) -> String {
        "hamming".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> f64 {
        x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count().as_f64()
    }

    fn is_expensive(&self) -> bool {
        self.is_expensive
    }
}

/// 1 - jaccard-similarity.
///
/// Warning: DO NOT use this with floating-point numbers.
#[derive(Debug)]
pub struct Jaccard {
    pub is_expensive: bool,
}

impl<T: Number> Metric<T> for Jaccard {
    fn name(&self) -> String {
        "jaccard".to_string()
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> f64 {
        if x.is_empty() || y.is_empty() {
            1.
        } else {
            let x = std::collections::HashSet::<u64>::from_iter(x.iter().map(|&a| NumCast::from(a).unwrap()));
            let y = std::collections::HashSet::from_iter(y.iter().map(|&a| NumCast::from(a).unwrap()));

            let intersection = x.intersection(&y).count();

            if intersection == x.len() && intersection == y.len() {
                1.
            } else {
                let union = x.union(&y).count();
                1. - intersection.as_f64() / union.as_f64()
            }
        }
    }

    fn is_expensive(&self) -> bool {
        self.is_expensive
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

        let metric = metric_from_name("euclideansq", false).unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 9.);

        let metric = metric_from_name("euclidean", false).unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 3.);

        let metric = metric_from_name("manhattan", false).unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 5.);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        let _ = metric_from_name::<f32>("aloha", false).unwrap();
    }
}
