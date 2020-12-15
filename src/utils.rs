use std::cmp::Ordering;
use std::env;
use std::path::PathBuf;

use ndarray::{Array1, Array2};
use ndarray_npy::{read_npy, ReadNpyError};

use crate::metric::Real;
use crate::types::Index;

// TODO: Implement function to download datasets from internet

pub static DATASETS: &[&str] = &[
    "annthyroid",  // 0
    "arrhythmia",  // 1
    "breastw",  // 2
    "cardio",  // 3
    "cover",  // 4
    "glass",  // 5
    "http",  // 6
    "ionosphere",  // 7
    "lympho",  // 8
    "mammography",  // 9
    "mnist",  // 10
    "musk",  // 11
    "optdigits",  // 12
    "pendigits",  // 13
    "pima",  // 14
    "satellite",  // 15
    "satimage-2",  // 16
    "shuttle",  // 17
    "smtp",  // 18
    "thyroid",  // 19
    "vertebral",  // 20
    "vowels",  // 21
    "wbc",  // 22
    "wine",  // 23
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

pub fn read_data_f64(dataset: &str) -> Result<(Array2<f64>, Array1<u8>), ReadNpyError> {
    let (data_path, labels_path) = get_data_paths(dataset).unwrap();
    let data: Array2<f64> = read_npy(data_path)?;
    let labels: Array1<u8> = read_npy(labels_path)?;
    Ok((data, labels))
}

pub fn read_apogee() -> Array2<f32> {
    // let mut data_dir: PathBuf = env::current_dir().unwrap();
    // data_dir.push("data");
    let mut data_dir: PathBuf = PathBuf::new();
    data_dir.push("/data");
    data_dir.push("nishaq");
    data_dir.push("APOGEE");
    data_dir.push("apogee_full.npy");
    // println!("{:?}", data_dir);
    read_npy(data_dir).unwrap()
}

//noinspection DuplicatedCode
#[allow(clippy::ptr_arg)]
pub fn argmin<T: Real>(values: &Vec<T>) -> (Index, T) {
    values.iter()
        .enumerate()
        .fold((0, values[0]), |(i_min, v_min), (i, &v)| {
            match v_min.partial_cmp(&v).unwrap() {
                Ordering::Less => (i_min, v_min),
                _ => (i, v)
            }
        })
}

//noinspection DuplicatedCode
#[allow(clippy::ptr_arg)]
pub fn argmax<T: Real>(values: &Vec<T>) -> (Index, T) {
    values.iter()
        .enumerate()
        .fold((0, values[0]), |(i_max, v_max), (i, &v)| {
            match v_max.partial_cmp(&v).unwrap() {
                Ordering::Greater => (i_max, v_max),
                _ => (i, v)
            }
        })
}


#[cfg(test)]
mod tests {
    use ndarray_npy::ReadNpyError;

    use crate::utils::{DATASETS, read_data_f64};

    #[test]
    fn test_read_data() -> Result<(), ReadNpyError> {
        read_data_f64(DATASETS[0])?;
        Ok(())
    }
}
