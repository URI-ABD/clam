//! Test the rust code.

use std::fs::read_dir;

use ndarray::s;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_array() {
        let chunk_along = 0;
        let size = 3;
        let shape = [100, 100, 100];

        // The number of files is the number of chunks + 1 for the metadata
        let filenum = shape[chunk_along] / size + 1;
        let dirname = "test_chunked_arr_large";

        let (tmp, _) = generate_dummy_data(dirname, chunk_along, size, &shape);

        // Assert that the correct number of chunks were written
        let files = read_dir(tmp.path()).unwrap().collect::<Vec<_>>();

        assert_eq!(filenum, files.len() - 1);

        let ca = ChunkedArray::<f32>::new(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(ca.chunked_along, chunk_along);
        assert_eq!(ca.chunk_size, size);
        assert_eq!(&ca.shape, &shape);
    }
    #[test]
    fn test_get_single_chunk() {
        let chunk_along = 0;
        let size = 3;
        let shape = [9, 3, 3];
        let dirname = "test_chunked_arr";

        let (tmp, arr) = generate_dummy_data(dirname, chunk_along, size, &shape);
        let ca = ChunkedArray::<f32>::new(tmp.path().to_str().unwrap()).unwrap();

        let slice = s![0..3, .., ..];
        let expected = arr.slice(slice.as_ref());
        let result = ca.get(slice.as_ref());

        assert_eq!(result, expected);
    }

    #[test]
    fn test_get_multiple_chunks() {
        let chunk_along = 0;
        let size = 3;
        let shape = [9, 3, 3];
        let dirname = "test_chunked_arr";

        let (tmp, arr) = generate_dummy_data(dirname, chunk_along, size, &shape);
        let ca = ChunkedArray::<f32>::new(tmp.path().to_str().unwrap()).unwrap();

        let slice = s![0..9, .., ..];
        let expected = arr.slice(slice.as_ref());
        let result = ca.get(slice.as_ref());

        assert_eq!(result, expected);
    }

    #[test]
    fn test_get_sliced() {
        let chunk_along = 0;
        let size = 3;
        let shape = [9, 3, 3];
        let dirname = "test_chunked_arr";

        let (tmp, arr) = generate_dummy_data(dirname, chunk_along, size, &shape);
        let ca = ChunkedArray::<f32>::new(tmp.path().to_str().unwrap()).unwrap();

        let slice = s![1..4, 1..3, 1..3];
        let expected = arr.slice(slice.as_ref());
        let result = ca.get(slice.as_ref());

        assert_eq!(expected, result);
    }

    #[test]
    fn test_get_random_slice() {
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
        let have = arr.slice(slice);
        let got = ca.get(slice.as_ref());

        assert_eq!(got.shape(), have.shape());
        assert_eq!(
            got.iter().collect::<Vec<&f32>>(),
            have.iter().collect::<Vec<&f32>>()
        );
    }

    #[test]
    fn test_get_random_slice_stepped() {
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
            rng.gen_range(0..shape[0] / 2)..rng.gen_range(shape[0] / 2..shape[0]);rng.gen_range(1..shape[0]),
            rng.gen_range(0..shape[1] / 2)..rng.gen_range(shape[1] / 2..shape[1]);rng.gen_range(1..shape[1]),
            rng.gen_range(0..shape[2] / 2)..rng.gen_range(shape[2] / 2..shape[2]);rng.gen_range(1..shape[2]),
        ];
        // Assert they're equal
        let have = arr.slice(slice);
        let got = ca.get(slice.as_ref());

        assert_eq!(got.shape(), have.shape());
        assert_eq!(
            got.iter().collect::<Vec<&f32>>(),
            have.iter().collect::<Vec<&f32>>()
        );
    }
}
