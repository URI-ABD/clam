// use std::path::PathBuf;

// use ndarray::prelude::*;
// use ndarray_npy::read_npy;
// use ndarray_npy::ReadNpyError;

// use clam::prelude::*;

// TODO: Implement function to download datasets from internet

// TODO: Discriminate by data type of downloaded data
// pub static SEARCH_DATASETS: &[(&str, &str)] = &[
//     ("deep-image", "cosine"),       // 0
//     ("fashion-mnist", "euclidean"), // 1
//     ("gist", "euclidean"),          // 2
//     ("glove-25", "cosine"),         // 3
//     ("glove-50", "cosine"),         // 4
//     ("glove-100", "cosine"),        // 5
//     ("glove-200", "cosine"),        // 6
//     ("kosarak", "jaccard"),         // 7
//     ("mnist", "euclidean"),         // 8
//     ("nytimes", "cosine"),          // 9
//     ("sift", "euclidean"),          // 10,
//     ("lastfm", "cosine"),           // 11
// ];

// TODO: Add sub-sampling and normalization

// pub type TrainTest<T> = (Vec<Vec<T>>, Vec<Vec<T>>);

// pub fn read_ann_data<T: Number, U: Number>(name: &str) -> Result<TrainTest<T>, ReadNpyError> {
//     let mut data_dir: PathBuf = PathBuf::new();
//     data_dir.push("/data");
//     data_dir.push("abd");
//     data_dir.push("ann_data");

//     let mut train_path = data_dir.clone();
//     train_path.push(format!("{:}-train.npy", name));
//     data_dir.push(format!("{:}-test.npy", name));

//     let train: Array2<T> = read_npy(train_path)?;
//     let train = train.outer_iter().map(|row| row.to_vec()).collect();

//     let test: Array2<T> = read_npy(data_dir)?;
//     let test = test.outer_iter().map(|row| row.to_vec()).collect();

//     Ok((train, test))
// }
