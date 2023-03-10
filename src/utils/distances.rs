use std::f64::EPSILON;

use crate::core::number::Number;

pub fn from_name<T: Number, U: Number>(name: &str) -> fn(&[T], &[T]) -> U {
    match name {
        "euclidean" => euclidean,
        "euclidean_sq" => euclidean_sq,
        "manhattan" => manhattan,
        "cosine" => cosine,
        "hamming" => hamming,
        "jaccard" => jaccard,
        _ => panic!("Distance {name} is not implemented in clam."),
    }
}

pub fn euclidean<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    U::from(euclidean_sq::<T, f64>(x, y).sqrt()).unwrap()
}

pub fn euclidean_sq<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| a - b).map(|v| v * v).sum();
    U::from(d).unwrap()
}

pub fn manhattan<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let d: T = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| if a > b { a - b } else { b - a })
        .sum();
    U::from(d).unwrap()
}

pub fn cosine<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let [xx, yy, xy] = x.iter().zip(y.iter()).fold([T::zero(); 3], |[xx, yy, xy], (&a, &b)| {
        [xx + a * a, yy + b * b, xy + a * b]
    });
    let [xx, yy, xy] = [xx.as_f64(), yy.as_f64(), xy.as_f64()];

    if xx <= EPSILON || yy <= EPSILON || xy <= EPSILON {
        U::one()
    } else {
        let d = 1. - xy / (xx * yy).sqrt();
        if d < EPSILON {
            U::zero()
        } else {
            U::from(d).unwrap()
        }
    }
}

pub fn hamming<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    U::from(x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count()).unwrap()
}

pub fn jaccard<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    if x.is_empty() || y.is_empty() {
        return U::one();
    }

    let x = std::collections::HashSet::<u64>::from_iter(x.iter().map(|a| a.as_u64()));
    let y = std::collections::HashSet::from_iter(y.iter().map(|a| a.as_u64()));

    let intersection = x.intersection(&y).count();

    if intersection == x.len() && intersection == y.len() {
        U::zero()
    } else {
        let union = x.union(&y).count();
        U::from(1. - intersection.as_f64() / union.as_f64()).unwrap()
    }
}
