//! Test the rust code.

use std::fs::read_dir;

use ndarray::{s, Array0, SliceInfoElem};
use ndarray::{Array, IxDyn};
use ndarray_chunked::chunked_array::{ChunkSettings, ChunkedArray};
use rand::Rng;
use tempdir::TempDir;

fn generate_dummy_data(
    dirname: &str,
    chunk_along: usize,
    size: usize,
    shape: &[usize],
) -> (TempDir, Array<f32, IxDyn>) {
    // Dummy array
    let arr = Array::range(0., shape.iter().fold(1., |acc, x| acc * (*x as f32)), 1.)
        .into_shape(shape)
        .unwrap();

    // Create our temp directory and settings
    let tmp = TempDir::new(dirname).unwrap();
    let settings = ChunkSettings::new(chunk_along, size, tmp.path().to_str().unwrap());

    // Write out the chunks
    ChunkedArray::chunk(&arr, &settings).unwrap();

    (tmp, arr)
}

#[test]
fn test_simple_array() {
    let chunk_along = 0;
    let size = 3;
    let shape = [9, 3, 3];
    let filenum = shape[chunk_along] / size + 1;
    let dirname = "test_chunked_arr";

    let (tmp, arr) = generate_dummy_data(dirname, chunk_along, size, &shape);

    // Assert that the correct number of chunks were written
    let files = read_dir(tmp.path()).unwrap().collect::<Vec<_>>();

    assert_eq!(filenum, files.len());

    let ca = ChunkedArray::<f32>::new(tmp.path().to_str().unwrap()).unwrap();
    assert_eq!(ca.chunked_along, chunk_along);
    assert_eq!(ca.chunk_size, size);
    assert_eq!(&ca.shape, &shape);

    // Smoke test
    assert_eq!(arr, ca.get(s![.., .., ..].as_ref()));

    // A randomized slice of our 3d array
    let mut rng = rand::thread_rng();

    let slice = s![
        rng.gen_range(0..shape[0] / 2)..rng.gen_range(shape[0] / 2..shape[0]),
        rng.gen_range(0..shape[1] / 2)..rng.gen_range(shape[1] / 2..shape[1]),
        rng.gen_range(0..shape[2] / 2)..rng.gen_range(shape[2] / 2..shape[2]),
    ];
    // Assert they're equal
    // assert_eq!(arr.slice(rand_slice), ca.get(rand_slice.as_ref())[0]);
    let have = arr.slice(slice);

    let got = ca.get(slice.as_ref());

    // assert_eq!(got.ndim(), 0);
    assert_eq!(got.shape(), have.shape());
    assert_eq!(
        got.iter().collect::<Vec<&f32>>(),
        have.iter().collect::<Vec<&f32>>()
    );
}

#[test]
fn test_large_array() {
    let chunk_along = 0;
    let size = 3;
    let shape = [100, 100, 100];
    let filenum = shape[chunk_along] / size + 1;
    let dirname = "test_chunked_arr_large";

    let (tmp, arr) = generate_dummy_data(dirname, chunk_along, size, &shape);

    // Assert that the correct number of chunks were written
    let files = read_dir(tmp.path()).unwrap().collect::<Vec<_>>();

    assert_eq!(filenum, files.len());

    let ca = ChunkedArray::<f32>::new(tmp.path().to_str().unwrap()).unwrap();
    assert_eq!(ca.chunked_along, chunk_along);
    assert_eq!(ca.chunk_size, size);
    assert_eq!(&ca.shape, &shape);

    println!("{:?}", ca.get(s![8, 2, ..].as_ref()));
}
