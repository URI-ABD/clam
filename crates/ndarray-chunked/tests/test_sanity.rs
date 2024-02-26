//! Test the rust code.

use std::fs::read_dir;

use ndarray::Array;
use ndarray_chunked::chunked_array::{ChunkSettings, ChunkedArray};
use tempdir::TempDir;

#[test]
fn test_write_simple_array() {
    let chunk_along = 2;
    let size = 1;
    let shape = [3, 3, 3];
    let filenum = shape[chunk_along] / size + 1;

    // Dummy array
    let arr = Array::range(0., shape.iter().fold(1., |acc, x| acc * (*x as f32)), 1.)
        .into_shape(shape)
        .unwrap();

    // Create our temp directory and settings
    let tmp = TempDir::new("test_chunked_arr").unwrap();
    let settings = ChunkSettings::new(chunk_along, size, tmp.path().to_str().unwrap());

    // Write out the chunks
    ChunkedArray::chunk(&arr, &settings).unwrap();

    // Assert that the correct number of chunks were written
    let files = read_dir(tmp.path()).unwrap().collect::<Vec<_>>();

    assert_eq!(filenum, files.len());

    let ca = ChunkedArray::new(tmp.path().to_str().unwrap()).unwrap();
    assert_eq!(ca.chunked_along, chunk_along);
    assert_eq!(ca.chunk_size, size);
    assert_eq!(&ca.shape, &shape)
}
