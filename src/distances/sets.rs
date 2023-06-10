use crate::core::number::Number;

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
        U::from(1. - intersection as f64 / union as f64).unwrap()
    }
}
