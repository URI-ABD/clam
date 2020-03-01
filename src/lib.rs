mod cluster;
mod dataset;
mod manifold;

pub mod criteria;
pub mod types;

pub use crate::manifold::Manifold;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::*;
    use types::{Data};
    use crate::dataset::Dataset;

    type Arr = Array2<f32>;

    fn line(n: usize, m: f32, b: f32) -> Arr {
        let mut arr = Arr::zeros((n, 2));
        // Build Xs
        let mut pointer = arr.column_mut(0);
        let mut xs = Array::range(0., n as f32, 1.);
        pointer = xs.view_mut(); 
        // Build Yx 
        let mut pointer = arr.column_mut(1);
        let mut ys = m * xs + b;
        pointer = ys.view_mut();

        arr
    }

    #[test]
    fn test_line() {
        let line = line(8, 0., 0.);
        let data = Data::from(line);
        let dataset = Dataset::new(data, "euclidean");
        let criteria = vec![criteria::MinPoints::new(1)];
        let manifold = Manifold::new(dataset, criteria);

        assert_eq!(manifold.leaves(Some(3)).len(), 8);
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
