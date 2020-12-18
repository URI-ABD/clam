use std::fmt::{Debug, Display};
use std::iter::Sum;

use ndarray::{ArrayView1, Zip};
use ndarray::parallel::prelude::*;
use num_traits::cast::FromPrimitive;
use num_traits::real::Real as _Real;

pub trait Real: _Real + FromPrimitive + Sum + Debug + Display + Send + Sync {}

impl Real for f32 {}
impl Real for f64 {}

pub type Metric<T, U> = fn(ArrayView1<T>, ArrayView1<T>) -> U;

pub fn metric_on_real<T: Real, U: Real>(metric: &'static str) -> Result<Metric<T, U>, String> {
    match metric {
        "euclidean" => Ok(euclidean),
        "par_euclidean" => Ok(par_euclidean),
        "euclideansq" => Ok(euclideansq),
        "par_euclideansq" => Ok(par_euclideansq),
        "manhattan" => Ok(manhattan),
        "par_manhattan" => Ok(par_manhattan),
        "cosine" => Ok(cosine),
        _ => Err(format!("{} is not defined.", metric)),
    }
}

fn euclidean<T: Real, U: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> U {
    euclideansq::<T, U>(x, y).sqrt()
}

fn par_euclidean<T: Real, U: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> U {
    par_euclideansq::<T, U>(x, y).sqrt()
}

fn euclideansq<T: Real, U: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> U {
    let d: T = x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    U::from(d).unwrap()
}

fn par_euclideansq<T: Real, U: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> U {
    let d: T = Zip::from(x)
        .and(y)
        .into_par_iter()
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    U::from(d).unwrap()
}

fn manhattan<T: Real, U: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> U {
    let d: T = x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();
    U::from(d).unwrap()
}

fn par_manhattan<T: Real, U: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> U {
    let d: T = Zip::from(x)
        .and(y)
        .into_par_iter()
        .map(|(&a, &b)| (a - b).abs())
        .sum();
    U::from(d).unwrap()
}

fn dot<T: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> T {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

fn cosine<T: Real, U: Real>(x: ArrayView1<T>, y: ArrayView1<T>) -> U {
    let xx = dot(x, x);
    if xx == T::zero() { return U::one(); }

    let yy = dot(y, y);
    if yy == T::zero() { return U::one(); }

    let xy = dot(x, y);
    if xy <= T::zero() { return U::one() }

    let similarity = U::from(xy * xy / (xx * yy)).unwrap();
    U::one() - similarity.sqrt()
}


#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::{arr2, Array2};

    use crate::metric::metric_on_real;

    #[test]
    fn test_on_real() {
        let data: Array2<f64> = arr2(&[[1., 2., 3.], [3., 3., 1.]]);

        let distance = metric_on_real("euclideansq").unwrap();
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 9.);

        let distance = metric_on_real("euclidean").unwrap();
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 3.);

        let distance = metric_on_real("manhattan").unwrap();
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 5.);
    }

    #[test]
    fn test_panic() {
        let f = metric_on_real::<f32, f32>("aloha");
        match f {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        };
    }
}
