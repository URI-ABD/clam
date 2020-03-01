mod cluster;
mod dataset;
mod manifold;

pub mod criteria;
pub mod types;

pub use crate::manifold::Manifold;

#[cfg(test)]
mod tests {
    use super::*;
    use types::*;
    use ndarray::{ArrayBase, Array2, array};
    use crate::dataset::Dataset;

    #[test]
    fn test_line() {
        // let xs = Array1::<f32>::zeros(8);
        // let ys = array![[0f32..8f32]];
        // let line: Array2<f32> = xs.iter().zip(ys.iter()).map(|(x, y)| (x, y)).collect();

        let mut line = Array2::<f32>::zeros((8, 2));
        let ys = ArrayBase::<f32>::range(0, 8, 1);
        line.column_mut(1) = ys;
        let data = Data::from(line);
        let dataset = Dataset::new(data, "euclidean");
        let criteria = vec![criteria::MinPoints::new(1)];
        let manifold = Manifold::new(dataset, criteria);

        assert_eq!(manifold.leaves(Some(3)), 8);
    }

    #[test]
    fn test_working() {
        let data = Data::from(array![[0, 0], [0, 1]]);
        let metric = "euclidean";
        let dataset = Dataset::new(data, metric);
        let criteria = vec![criteria::MinPoints::new(2)];
        let manifold = Manifold::new(dataset, criteria);
    }
}
