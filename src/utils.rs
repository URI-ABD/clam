use std::path::PathBuf;

use ndarray::prelude::*;
use ndarray_npy::read_npy;
use ndarray_npy::ReadNpyError;

use crate::prelude::*;

// TODO: Implement function to download datasets from internet

pub static CHAODA_DATASETS: &[&str] = &[
    "annthyroid",  // 0
    "arrhythmia",  // 1
    "breastw",     // 2
    "cardio",      // 3
    "cover",       // 4
    "glass",       // 5
    "http",        // 6
    "ionosphere",  // 7
    "lympho",      // 8
    "mammography", // 9
    "mnist",       // 10
    "musk",        // 11
    "optdigits",   // 12
    "pendigits",   // 13
    "pima",        // 14
    "satellite",   // 15
    "satimage-2",  // 16
    "shuttle",     // 17
    "smtp",        // 18
    "thyroid",     // 19
    "vertebral",   // 20
    "vowels",      // 21
    "wbc",         // 22
    "wine",        // 23
];

// TODO: Discriminate by data type of downloaded data
pub static ANN_DATASETS: &[(&str, &str)] = &[
    ("deep-image", "cosine"),       // 0
    ("fashion-mnist", "euclidean"), // 1
    ("gist", "euclidean"),          // 2
    ("glove-25", "cosine"),         // 3
    ("glove-50", "cosine"),         // 4
    ("glove-100", "cosine"),        // 5
    ("glove-200", "cosine"),        // 6
    ("kosarak", "jaccard"),         // 7
    ("mnist", "euclidean"),         // 8
    ("nytimes", "cosine"),          // 9
    ("sift", "euclidean"),          // 10,
    ("lastfm", "cosine"),           // 11
];

// TODO: Add sub-sampling and normalization

pub type DataLabels<T, U> = (Vec<Vec<T>>, Vec<U>);
pub type TrainTest<T> = (Vec<Vec<T>>, Vec<Vec<T>>);

pub fn read_test_data() -> DataLabels<f64, u8> {
    let mut data_dir: PathBuf = std::env::current_dir().unwrap();
    data_dir.push("data");

    let mut data_path = data_dir.clone();
    data_path.push("annthyroid.npy");
    data_dir.push("annthyroid_labels.npy");

    let data: Array2<f64> = read_npy(data_path).unwrap();
    let data = data.outer_iter().map(|row| row.to_vec()).collect();

    let labels: Array1<u8> = read_npy(data_dir).unwrap();

    (data, labels.to_vec())
}

pub fn read_chaoda_data(
    path: PathBuf,
    read_labels: bool,
) -> Result<DataLabels<f64, bool>, String> {
    let mut data_path = path.clone();
    data_path.set_extension("npy");

    let data: Array2<f64> = read_npy(data_path.clone()).map_err(|error| {
        format!(
            "Error: Failed to read your dataset at {}. {:}",
            data_path.to_str().unwrap(),
            error
        )
    })?;
    let data: Vec<_> = data.outer_iter().map(|row| row.to_vec()).collect();

    let labels = if read_labels {
        let mut labels_path = path.clone();
        labels_path.pop();
        let name = path.file_name().unwrap().to_str().unwrap();
        labels_path.push(format!("{}_labels.npy", name));
        let labels: Array1<u8> =
            read_npy(labels_path.clone()).map_err(|error| {
                format!(
                    "Error: Failed to read your dataset at {}. {:}",
                    labels_path.to_str().unwrap(),
                    error
                )
            })?;
        labels.into_iter().map(|label| label != 0).collect()
    } else {
        data.iter().map(|_| true).collect()
    };

    Ok((data, labels))
}

pub fn read_ann_data<T: Number, U: Number>(
    name: &str,
) -> Result<TrainTest<T>, ReadNpyError> {
    let mut data_dir: PathBuf = PathBuf::new();
    data_dir.push("/data");
    data_dir.push("abd");
    data_dir.push("ann_data");

    let mut train_path = data_dir.clone();
    train_path.push(format!("{:}-train.npy", name));
    data_dir.push(format!("{:}-test.npy", name));

    let train: Array2<T> = read_npy(train_path)?;
    let train = train.outer_iter().map(|row| row.to_vec()).collect();

    let test: Array2<T> = read_npy(data_dir)?;
    let test = test.outer_iter().map(|row| row.to_vec()).collect();

    Ok((train, test))
}

pub fn argmin<T: Number>(values: &[T]) -> (Index, T) {
    values
        .iter()
        .enumerate()
        .fold((0, values[0]), |(i_min, v_min), (i, &v)| {
            if v < v_min {
                (i, v)
            } else {
                (i_min, v_min)
            }
        })
}

pub fn argmax<T: Number>(values: &[T]) -> (Index, T) {
    values
        .iter()
        .enumerate()
        .fold((0, values[0]), |(i_max, v_max), (i, &v)| {
            if v > v_max {
                (i, v)
            } else {
                (i_max, v_max)
            }
        })
}
