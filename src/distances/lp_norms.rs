use crate::core::number::Number;

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

/// Lebesgue L3 norm
pub fn l3_norm<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let d: T = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| if a > b { a - b } else { b - a })
        .map(|v| v * v * v)
        .sum();
    let d = d.as_f64().cbrt();
    U::from(d).unwrap()
}

pub fn l4_norm<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let d: T = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| a - b)
        .map(|v| v * v * v * v)
        .sum();
    let d = d.as_f64().sqrt().sqrt();
    U::from(d).unwrap()
}
