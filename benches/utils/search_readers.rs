#![allow(dead_code, unused_imports, unused_variables)]

use ndarray::prelude::*;

use clam::prelude::*;
pub type TrainTest<T> = (Vec<Vec<T>>, Vec<Vec<T>>);

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

pub fn read_search_data<T: Number>(name: &str) -> Result<TrainTest<T>, String> {
    todo!()
}
