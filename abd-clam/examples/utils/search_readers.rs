use ndarray::prelude::*;

type TrainTest<T> = (Vec<Vec<T>>, Vec<Vec<T>>);

pub static SEARCH_DATASETS: &[(&str, &str)] = &[
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
    ("sift", "euclidean"),          // 10
    ("lastfm", "cosine"),           // 11
];

fn make_path(dir: &std::path::Path, name: &str, variant: &str) -> std::path::PathBuf {
    let mut path = dir.to_path_buf();
    path.push("as_npy");
    path.push(format!("{name}_{variant}.npy"));
    assert!(path.exists(), "Path not found: {path:?}");
    path
}

fn read_npy(path: &std::path::PathBuf) -> Result<Vec<Vec<f32>>, String> {
    let data: Array2<f32> = ndarray_npy::read_npy(path).map_err(|error| {
        format!(
            "Error: Failed to read your dataset at {}. {:}",
            path.to_str().unwrap(),
            error
        )
    })?;

    Ok(data.outer_iter().map(|row| row.to_vec()).collect())
}

pub fn read_search_data(name: &str) -> Result<TrainTest<f32>, String> {
    let mut data_dir = std::env::current_dir().unwrap();
    data_dir.pop();
    data_dir.push("data");
    data_dir.push("search_small");

    assert!(data_dir.exists(), "Path not found: {data_dir:?}");

    let train_data = read_npy(&make_path(&data_dir, name, "train"))?;

    let test_data = read_npy(&make_path(&data_dir, name, "test"))?;

    Ok((train_data, test_data))
}
