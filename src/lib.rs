mod cluster;
mod dataset;
mod manifold;

pub mod criteria;
pub mod types;

pub use crate::manifold::Manifold;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working() {
        let data = Box::new(vec![1, 2, 3]);
        let metric = String::from("euclidean");
        Manifold::new(data, metric, vec![criteria::MinPoints::new(2)]);
    }
}
