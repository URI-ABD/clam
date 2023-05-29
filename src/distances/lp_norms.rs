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
