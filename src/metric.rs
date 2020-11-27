use ndarray::ArrayView1;

// TODO: Parallelize these functions using ndarray's parallelism interface, and avoid using .to_vec()

fn euclidean(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    euclideansq(x, y).sqrt()
}

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

#[derive(Debug)]
pub struct Metric;

impl Metric {
    // TODO: remove panic! and return Result instead.
    pub fn on_f64(metric: &'static str) -> fn(ArrayView1<f64>, ArrayView1<f64>) -> f64 {
        match metric {
            "euclidean" => euclidean,
            "euclideansq" => euclideansq,
            "manhattan" => manhattan,
            _ => panic!("{} is not defined.", metric),
        }
    }
}


#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::{Array2, arr2};

    use super::Metric;

    #[test]
    fn test_builder() {
        let data: Array2<f64> = arr2(&[[1., 2., 3.], [3., 3., 1.]]);

        let distance = Metric::on_f64("euclideansq");
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 9.);

        let distance = Metric::on_f64("euclidean");
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 3.);

        let distance = Metric::on_f64("manhattan");
        approx_eq!(f64, distance(data.row(0), data.row(0)), 0.);
        approx_eq!(f64, distance(data.row(0), data.row(1)), 5.);
    }

    #[test]
    fn test_debug(){ assert_eq!(format!("The metric is: {:?}", Metric), "The metric is: Metric"); }

    #[test]
    #[should_panic]
    fn test_panic() { Metric::on_f64("aloha"); }
}
