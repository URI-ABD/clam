use std::{
    fmt::Debug,
    fs::{create_dir, File},
    io::{Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
};

use ndarray::{
    concatenate, Array, ArrayBase, Axis, Dimension, IxDyn, OwnedRepr, SliceInfo, SliceInfoElem,
};

use ndarray_npy::{read_npy, write_npy, ReadableElement, WritableElement};

///.
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
fn slice_info_to_vec(info: &SliceInfoElem, end: usize) -> Vec<usize> {
    match info {
        // Need to account for negative indexing
        SliceInfoElem::Slice {
            start,
            end: stop,
            step: _,
        } => {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
            let adjusted_start = if *start >= 0 {
                // This conversion is fine because we require it to be nonnegative to get here
                (*start) as usize
            } else {
                // This conversion is also fine because this value is assumed to be less than `-end`.
                (end as isize + *start) as usize
            };

            let adjusted_stop: usize = stop.as_ref().map_or(end, |s| {
                // Again, these are fine
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                if *s >= 0 {
                    *s as usize
                } else {
                    (end as isize + s) as usize
                }
            });

            // TODO: Negative stepping
            #[allow(clippy::cast_sign_loss)]
            (adjusted_start..adjusted_stop).collect()
        }

        SliceInfoElem::Index(x) => {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
            let x = if *x >= 0 {
                *x as usize
            } else {
                (end as isize + x) as usize
            };
            vec![x]
        }
        SliceInfoElem::NewAxis => todo!(),
    }
}

///
pub struct ChunkSettings {
    /// The axis we'll chunk along
    chunk_along: usize,
    /// The size (number of indices along the dimesion) of each chunk
    size: usize,
    /// The path of the containing folder
    path: String,
}

impl ChunkSettings {
    /// Constructs a new `ChunkSettings`
    #[must_use]
    pub fn new(chunk_along: usize, size: usize, path: &str) -> Self {
        Self {
            chunk_along,
            size,
            path: path.to_string(),
        }
    }
}

/// The `ChunkedArray` data structure. Currently without any caching
pub struct ChunkedArray<T: ReadableElement> {
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

impl<T: ReadableElement + WritableElement + Clone + Default + Debug> ChunkedArray<T> {
    /// Returns a given slice from the `ChunkedArray`
    /// # Panics
    /// This function will panic if chunk files are malformed or if conversions break
    /// TODO: Improve these docs
    #[must_use]
    #[allow(clippy::unwrap_used)]
    pub fn get(&self, idxs: &[SliceInfoElem]) -> Array<T, IxDyn> {
        let mut idxs = idxs.to_owned();

        // Now we need to resolve which chunks we actually need
        let info = idxs[self.chunked_along];
        let slice_axis_indices: Vec<usize> =
            slice_info_to_vec(&info, self.shape[self.chunked_along]);

        // Before we load in the chunks we first need to calculate which chunks we'll even need
        let start_chunk = slice_axis_indices[0] / self.chunk_size;
        let end_chunk = slice_axis_indices[slice_axis_indices.len() - 1] / self.chunk_size;

        // Now load in the actual chunks
        let chunks = (start_chunk..=end_chunk).map(|n| {
            let path = self.path.join(format!("chunk{n}.npy"));
            read_npy(path).unwrap()
        });

        // This is just all our chunks concatenated together.
        let chunk: Option<ArrayBase<OwnedRepr<_>, _>> =
            chunks.into_iter().fold(None, |acc, c: Array<T, IxDyn>| {
                acc.map_or_else(
                    || Some(c.to_owned()),
                    |partial| {
                        let bound =
                            concatenate(Axis(self.chunked_along), &[partial.view(), c.view()])
                                .unwrap();
                        Some(bound)
                    },
                )
            });

        // TODO: Fix this. I used an option in the fold because its the simplest way of doing it
        let chunk = chunk.unwrap();

        // Align the chunked_along axis slicing info to match new `chunk`
        #[allow(clippy::match_on_vec_items, clippy::cast_possible_wrap)]
        let adjusted_chunk_info = match idxs[self.chunked_along] {
            SliceInfoElem::Slice { start, end, step } => {
                // We have to do some adjusting here. We need to adjust the start and end to be relative to the chunk
                // because the slice might not be perfect.
                let adj_start = start % (self.chunk_size as isize);

                // Originally, the range was relative to `start` but we're shifting it around
                // to reflect that our chunk might not start at `start`. I.e. end = start + k
                // to begin with, but now we want end = adj_start + k. To do this we just
                // subtract `start` to get `k` and then add `adj_start`.
                let adj_end = end.map(|e| e - start + adj_start);
                SliceInfoElem::Slice {
                    start: adj_start,
                    end: adj_end,
                    step,
                }
            }

            #[allow(clippy::cast_possible_wrap)]
            SliceInfoElem::Index(i) => SliceInfoElem::Index(i % (self.chunk_size as isize)),
            SliceInfoElem::NewAxis => todo!(),
        };

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
        //ndim; shape; axis along which the array was split; the max size of each chunk
        let (shape, chunked_along, chunk_size) = Self::load_metadata(folder)?;

        Ok(Self {
            chunked_along,
            chunk_size,
            shape,
            path: folder.to_owned(),
            _t: PhantomData,
        })
    }

    /// Loads in `ChunkedArray` metadata from a given directory
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

    /// Chunks and writes out a given array. Chunking is specified by a `ChunkSettings` instance.
    /// # Errors
    pub fn chunk<D: Dimension>(arr: &Array<T, D>, settings: &ChunkSettings) -> Result<(), String> {
        let path = Path::new(&settings.path);

        // Create the directory, etc.
        Self::handle_directory(path)?;

        // Basic settings
        let (chunk_along, size) = (settings.chunk_along, settings.size);

        // Write out metadata
        {
            let metdata_bytes = Self::generate_metadata(arr, chunk_along, size);
            let mut handle =
                File::create(path.join(METADATA_FILENAME)).map_err(|e| e.to_string())?;
            handle
                .write_all(&metdata_bytes)
                .map_err(|e| e.to_string())?;
        }

        // Iterator of chunks along our specified axis of specified size along axis
        let chunk_iter = arr.axis_chunks_iter(Axis(chunk_along), size);

        // Write out each chunk
        for (ix, chunk) in chunk_iter.enumerate() {
            let chunk_path = path.join(format!("chunk{ix}.npy"));
            let chunk_path_name = chunk_path
                .to_str()
                .ok_or_else(|| "Could not convert path name to UTF-8".to_string())?;

            write_npy(chunk_path_name, &chunk).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Generates metadata file for a given array
    #[allow(clippy::cast_possible_truncation)]
    fn generate_metadata<A: WritableElement, D: Dimension>(
        arr: &Array<A, D>,
        chunk_along: usize,
        size: usize,
    ) -> Vec<u8> {
        // Format is simple. everything is taken as a u32
        // ndim; shape; axis along which the array was split; the max size of each chunk

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

            // At this point we know its a directory so we can just check if we're allowed to delete it
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
