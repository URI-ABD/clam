//! Test the rust code.

use std::fs::read_dir;

use ndarray::{s, Array, Dim, IxDyn};
use ndarray_chunked::chunked_array::{ChunkSettings, ChunkedArray};
use tempdir::TempDir;

fn generate_dummy_data(dirname: &str, chunk_along: usize, size: usize, shape: &[usize]) -> TempDir {
    // Dummy array
    let arr = Array::range(0., shape.iter().fold(1., |acc, x| acc * (*x as f32)), 1.)
        .into_shape(shape)
        .unwrap();

    // Create our temp directory and settings
    let tmp = TempDir::new(dirname).unwrap();
    let settings = ChunkSettings::new(chunk_along, size, tmp.path().to_str().unwrap());

    // Write out the chunks
    ChunkedArray::chunk(&arr, &settings).unwrap();

    tmp
}

#[test]
fn test_write_simple_array() {
    let chunk_along = 0;
    let size = 3;
    let shape = [9, 3, 3];
    let filenum = shape[chunk_along] / size + 1;
    let dirname = "test_chunked_arr";

    let tmp = generate_dummy_data(dirname, chunk_along, size, &shape);

    // Assert that the correct number of chunks were written
    let files = read_dir(tmp.path()).unwrap().collect::<Vec<_>>();

    assert_eq!(filenum, files.len());

    let ca = ChunkedArray::<f32>::new(tmp.path().to_str().unwrap()).unwrap();
    assert_eq!(ca.chunked_along, chunk_along);
    assert_eq!(ca.chunk_size, size);
    assert_eq!(&ca.shape, &shape);
    assert_eq!(ca.num_chunks(), 3);

    println!("{:?}", ca.get(s![.., .., 0].as_ref()));
}
