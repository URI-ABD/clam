use super::number::Number;

trait Instance<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    fn distance_to(&self, other: &Self) -> U;
}

// Implements Euclidean
impl<T: Number, U: Number> Instance<T, U> for Vec<T> {
    fn distance_to(&self, other: &Self) -> U {
        let d: T = self.iter().zip(other.iter()).map(|(&a, &b)| a - b).map(|v| v * v).sum();
        U::from(d.as_f64().sqrt()).unwrap()
    }
}
