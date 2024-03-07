use std::{
    fs::{create_dir, File},
    io::{Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
};

use ndarray::{
    concatenate, Array, ArrayBase, ArrayView, Axis, Dimension, IxDyn, OwnedRepr, SliceInfo,
    SliceInfoElem,
};
use ndarray_npy::{read_npy, write_npy, ReadableElement, WritableElement};

/// The filename for the metadata file of a `ChunkedArray`
static METADATA_FILENAME: &str = "metadata.bin";

/// Returns false if the given path is either not a directory or is a directory and is nonempty
fn directory_is_empty(path: &Path) -> Result<bool, String> {
    if path.is_dir() {
        // I.e. if the list of files in the directory is empty
        Ok(path.read_dir().map_err(|e| e.to_string())?.next().is_none())
    } else {
        Ok(false)
    }
}

/// Returns the intended list of indices specified by a `SliceInfoElem`
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn slice_info_to_vec(info: &SliceInfoElem, end: usize) -> Vec<usize> {
    let end = end as isize;

    match info {
        // Need to account for negative indexing
        SliceInfoElem::Slice {
            start,
            end: stop,
            step: _,
        } => {
            // Resolve indexing. If it's negative then it'll wrap around as expected
            let adjusted_start = start.rem_euclid(end);
            let adjusted_stop = stop.map_or(end, |e| e.rem_euclid(end + 1));

            // These casts are fine because rem_euclid is positive.
            ((adjusted_start as usize)..(adjusted_stop as usize)).collect()
        }

        SliceInfoElem::Index(x) => vec![x.rem_euclid(end) as usize],
        SliceInfoElem::NewAxis => todo!(),
    }
}

/// The `ChunkedArray` data structure. Currently without any caching
pub struct ChunkedArray<T> {
    /// The axis along which this Array was chunked along
    pub chunked_along: usize,
    /// The size of each chunk
    pub chunk_size: usize,
    /// The overall shape of the original array
    pub shape: Vec<usize>,
    /// Path to folder containing chunks
    path: PathBuf,
    /// Type of the data associated with the array
    _t: PhantomData<T>,
}

impl<T: Clone + ReadableElement + WritableElement> ChunkedArray<T> {
    /// Returns a given slice from the `ChunkedArray`
    /// # Panics
    /// This function will panic if chunk files are malformed or if conversions break
    /// TODO: Improve these docs
    #[must_use]
    #[allow(clippy::unwrap_used)]
    pub fn slice(&self, idxs: &[SliceInfoElem]) -> Array<T, IxDyn> {
        // Now we need to resolve which chunks we actually need
        let slice_axis_indices: Vec<usize> =
            slice_info_to_vec(&idxs[self.chunked_along], self.shape[self.chunked_along]);

        // Before we load in the chunks we first need to calculate which chunks we'll even need
        let start_chunk = slice_axis_indices[0] / self.chunk_size;

        // This is fine because we're guaranteed at least one element in `slice_axis_indices`
        let end_chunk = slice_axis_indices.last().unwrap() / self.chunk_size;

        // Now load in the actual chunks
        let chunks = (start_chunk..=end_chunk).map(|n| {
            let path = self.path.join(format!("chunk{n}.npy"));
            read_npy(path).unwrap()
        });

        // Concatenate the chunks
        let chunk = chunks
            .into_iter()
            .reduce(|a: ArrayBase<OwnedRepr<T>, _>, b| {
                concatenate(Axis(self.chunked_along), &[a.view(), b.view()]).unwrap()
            })
            .unwrap();

        // Align the chunked_along axis slicing info to match new `chunk`
        // Allowing this here is fine because both of the wraps only fail if chunk size is close to isize::MAX, which
        // is unlikely to happen.
        #[allow(clippy::cast_possible_wrap)]
        let adjusted_chunk_info = match idxs[self.chunked_along] {
            SliceInfoElem::Slice { start, end, step } => {
                // We have to do some adjusting here. We need to adjust the start and end to be relative to the chunk
                // because the slice might not be perfect and it might not start at 0. I.e. if your starting index is
                // `i` and you have chunks of size `k`, then i = n * k + r for some n and r. We have already resolved
                // `n` (by loading in the chunk `i` is in) so we just need `r` to resolve the correct index relative
                // to the chunk.
                let adj_start = start % (self.chunk_size as isize);

                // Then, we have to resolve the end index. Basically, end = start + k for some k but start has been
                // adjusted so we need to adjust end as well. We want to shift end by the same amount start was
                // shifted, so basically just adj_end = (start + k) - start + adj_start = adj_start + k shifts
                // end to the right spot.
                let adj_end = end.map(|e| e - start + adj_start);
                SliceInfoElem::Slice {
                    start: adj_start,
                    end: adj_end,
                    step,
                }
            }

            SliceInfoElem::Index(i) => SliceInfoElem::Index(i % (self.chunk_size as isize)),
            SliceInfoElem::NewAxis => todo!(),
        };

        // We'll take ownership here because we're going to be modifying the slice info
        let mut idxs = idxs.to_owned();

        // Align the axis beginning
        idxs[self.chunked_along] = adjusted_chunk_info;

        // Generate the slice info
        let sliceinfo: SliceInfo<_, IxDyn, IxDyn> =
            (idxs.as_ref() as &[SliceInfoElem]).try_into().unwrap();

        // Slice the chunk
        chunk.slice(sliceinfo).to_owned()
    }

    /// Attempts to create a new `ChunkedArray` from a given directory
    /// # Errors
    pub fn new(folder: &str) -> Result<Self, String> {
        let folder = Path::new(folder);

        // Handle metadata
        let (shape, chunked_along, chunk_size) = Self::load_metadata(folder)?;

        Ok(Self {
            chunked_along,
            chunk_size,
            shape,
            path: folder.to_owned(),
            _t: PhantomData,
        })
    }

    /// Loads in `ChunkedArray` metadata from a given directory.
    /// The format is simple: ndim; shape; axis along which the array was split; the max size of each chunk. Each
    /// data point is a u32 and is little endian. Shape is a list of u32s of length ndim.
    /// TODO: Complex nums
    ///
    /// # Errors
    fn load_metadata(folder: &Path) -> Result<(Vec<usize>, usize, usize), String> {
        let mut handle = File::open(folder.join(METADATA_FILENAME)).map_err(|e| e.to_string())?;

        // Buffer
        let mut u32_buf = [0_u8; 4];

        // Read ndim
        handle.read(&mut u32_buf).map_err(|e| e.to_string())?;
        let ndim = u32::from_le_bytes(u32_buf);

        // Read shape
        let mut shape: Vec<usize> = vec![];
        for _ in 0..ndim {
            handle.read(&mut u32_buf).map_err(|e| e.to_string())?;
            shape.push(u32::from_le_bytes(u32_buf) as usize);
        }

        // Read chunked_along
        handle.read(&mut u32_buf).map_err(|e| e.to_string())?;
        let chunked_along = u32::from_le_bytes(u32_buf) as usize;

        // Read chunk size
        handle.read(&mut u32_buf).map_err(|e| e.to_string())?;
        let chunk_size = u32::from_le_bytes(u32_buf) as usize;

        Ok((shape, chunked_along, chunk_size))
    }

    /// Chunks and writes out a given array
    /// # Errors
    pub fn chunk<D: Dimension>(
        arr: &ArrayView<'_, T, D>,
        chunk_along: usize,
        size: usize,
        path: &str,
    ) -> Result<(), String> {
        let path = Path::new(path);

        // Create the directory, etc.
        Self::handle_directory(path)?;

        // Write out metadata
        {
            let metadata_bytes = Self::generate_metadata(arr, chunk_along, size);
            let mut handle =
                File::create(path.join(METADATA_FILENAME)).map_err(|e| e.to_string())?;
            handle
                .write_all(&metadata_bytes)
                .map_err(|e| e.to_string())?;
        }

        // Iterator of chunks along our specified axis of specified size along axis
        let chunk_iter = arr.axis_chunks_iter(Axis(chunk_along), size);

        // Write out each chunk
        for (ix, chunk) in chunk_iter.enumerate() {
            let chunk_path = path.join(format!("chunk{ix}.npy"));
            let chunk_path_name = chunk_path.to_str().ok_or("Path is not valid UTF-8")?;
            write_npy(chunk_path_name, &chunk).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Generates metadata file for a given array
    #[allow(clippy::cast_possible_truncation)]
    fn generate_metadata<A: WritableElement, D: Dimension>(
        arr: &ArrayView<'_, A, D>,
        chunk_along: usize,
        size: usize,
    ) -> Vec<u8> {
        // Convert everything to u32 and then little endian bytes
        let ndim_bytes = (arr.ndim() as u32).to_le_bytes().to_vec();
        let shape_bytes = arr
            .shape()
            .iter()
            .flat_map(|a| (*a as u32).to_le_bytes().to_vec())
            .collect::<Vec<u8>>();
        let chunk_along_bytes = (chunk_along as u32).to_le_bytes().to_vec();
        let size_bytes = (size as u32).to_le_bytes().to_vec();

        // Concat all the bytes together
        [ndim_bytes, shape_bytes, chunk_along_bytes, size_bytes]
            .into_iter()
            .flatten()
            .collect()
    }

    /// Creates the new directory (if necesary) and handles basic errors
    fn handle_directory(path: &Path) -> Result<(), String> {
        if path.exists() {
            if !path.is_dir() {
                return Err("Path exists and is not a directory".to_string());
            }

            if !directory_is_empty(path)? {
                return Err("Directory exists".to_string());
            }
        } else {
            // If we're here then we can create the directory
            create_dir(path).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}
