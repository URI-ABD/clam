use std::env;
use std::path::PathBuf;

use ndarray::{Array1, Array2};
use ndarray_npy::{read_npy, ReadNpyError};

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

pub fn get_data_paths(dataset: &str) -> Result<(PathBuf, PathBuf), std::io::Error>{
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
