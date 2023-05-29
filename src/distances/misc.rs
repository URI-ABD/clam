use std::f64::EPSILON;

use crate::core::number::Number;

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
