use std::fmt::Debug;

use ndarray::ArrayView1;

#[derive(Debug)]
pub struct Metric;
pub type DistanceFunction = fn(ArrayView1<f64>, ArrayView1<f64>) -> f64;

impl Metric {
    pub fn on_float(metric: &'static str) -> Result<DistanceFunction, String> {
        match metric {
            "euclidean" => Ok(euclidean),
            "euclideansq" => Ok(euclideansq),
            "manhattan" => Ok(manhattan),
            _ => Err(format!("{} is not defined.", metric)),
        }
    }
}

fn euclidean(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 { euclideansq(x, y).sqrt() }

fn euclideansq(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum()
}

fn manhattan(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum()
}


#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::{arr2, Array2};

    use super::Metric;

    #[test]
    fn test_builder() {
        let data: Array2<f64> = arr2(&[[1., 2., 3.], [3., 3., 1.]]);

        let distance = Metric::on_float("euclideansq").unwrap();
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 9.);

        let distance = Metric::on_float("euclidean").unwrap();
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 3.);

        let distance = Metric::on_float("manhattan").unwrap();
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 5.);
    }

    #[test]
    fn test_debug(){ assert_eq!(format!("The metric is: {:?}", Metric), "The metric is: Metric"); }

    #[test]
    fn test_panic() {
        match Metric::on_float("aloha") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        };
    }
}
