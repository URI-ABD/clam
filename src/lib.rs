mod cluster;
mod dataset;
mod manifold;

pub mod criteria;
pub mod types;

pub use crate::manifold::Manifold;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn working() {
        let data = types::Data::from(array![[0, 0], [0, 1]]);
        let metric = String::from("euclidean");
        Manifold::new(Box::new(data), metric, vec![criteria::MinPoints::new(2)]);
    }
}
