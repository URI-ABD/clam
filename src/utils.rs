use std::env;
use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_npy::{read_npy, ReadNpyError};

use crate::types::Index;

pub static DATASETS: &[&str] = &[
    "annthyroid",
    "arrhythmia",
    "breastw",
    "cardio",
    "cover",
    "glass",
    "http",
    "ionosphere",
    "lympho",
    "mammography",
    "mnist",
    "musk",
    "optdigits",
    "pendigits",
    "pima",
    "satellite",
    "satimage-2",
    "shuttle",
    "smtp",
    "thyroid",
    "vertebral",
    "vowels",
    "wbc",
    "wine",
];

// TODO: Add subsampling and normalization

fn get_data_paths(dataset: &str) -> Result<(PathBuf, PathBuf), std::io::Error>{
    let mut data_dir: PathBuf = env::current_dir()?;
    data_dir.push("data");

    let mut data_path = data_dir.clone();
    data_path.push(format!("{}.npy", dataset));
    data_dir.push(format!("{}_labels.npy", dataset));
    Ok((data_path, data_dir))
}

pub fn read_data(dataset: &str) -> Result<(Array2<f64>, Array1<u8>), ReadNpyError> {
    let (data_path, labels_path) = get_data_paths(dataset).unwrap();
    let data: Array2<f64> = read_npy(data_path)?;
    let labels: Array1<u8> = read_npy(labels_path)?;
    Ok((data, labels))
}

pub fn argmin(values: ArrayView1<f64>) -> (Index, f64) {
    values
        .iter()
        .enumerate()
        .fold((0, values[0]), |(i_min, v_min), (i, v)| {
            if &v_min < v { (i_min, v_min) }
            else { (i, *v) }
        })
}

pub fn argmax(values: ArrayView1<f64>) -> (Index, f64) {
    values
        .iter()
        .enumerate()
        .fold((0, values[0]), |(i_max, v_max), (i, v)| {
            if &v_max > v { (i_max, v_max) }
            else { (i, *v) }
        })
}


#[cfg(test)]
mod tests {
    use ndarray_npy::ReadNpyError;

    use crate::utils::{DATASETS, read_data};

    #[test]
    fn test_read_data() -> Result<(), ReadNpyError> {
        for &dataset in DATASETS.iter() {
            let (data, labels) = read_data(dataset)?;
            println!("{:}: data-shape: {:?}, labels-shape {:?}", dataset, data.shape(), labels.shape());
        }
        Ok(())
    }
}
