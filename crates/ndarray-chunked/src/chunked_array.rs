use std::{
    fmt::Debug,
    fs::{create_dir, File},
    io::{Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    vec,
};

use ndarray::{Array, Axis, Dim, Dimension, IxDyn, SliceInfo, SliceInfoElem};
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

/// Returns the intended list of indices specified by a SliceInfoElem
fn slice_info_to_vec(info: &SliceInfoElem, end: usize) -> Vec<usize> {
    match info {
        // Need to account for negative indexing
        SliceInfoElem::Slice {
            start,
            end: stop,
            step,
        } => {
            // TODO: This can be fucked up and negative and it shouldnt be
            let adjusted_start = if *start >= 0 {
                (*start) as usize
            } else {
                (end as isize + *start) as usize
            };

            // TODO: Again
            let adjusted_stop: usize = match stop {
                Some(s) => {
                    if *s >= 0 {
                        *s as usize
                    } else {
                        (end as isize + s) as usize
                    }
                }
                None => end,
            };

            (adjusted_start..adjusted_stop)
                .step_by(*step as usize)
                .collect()
        }
        SliceInfoElem::Index(x) => {
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

///.
pub struct ChunkSettings {
    ///.
    chunk_along: usize,
    ///.
    size: usize,
    ///.
    path: String,
    ///.
    delete_dir: bool,
}

impl ChunkSettings {
    /// .
    #[must_use]
    pub fn new(chunk_along: usize, size: usize, path: &str) -> Self {
        Self {
            chunk_along,
            size,
            path: path.to_string(),
            delete_dir: false,
        }
    }

    /// Warning. Only set this to true you are sure the directory you're pointing to is safe to delete
    /// Turning this on will delete the directory at `path` if it already exists when writing the new array.
    pub fn set_delete_dir(&mut self, value: bool) {
        self.delete_dir = value;
    }
}

/// .
pub struct ChunkedArray<T: ReadableElement, D: Dimension> {
    /// The axis along which this Array was chunked along
    pub chunked_along: usize,
    /// The size of each chunk
    pub chunk_size: usize,
    /// The overall shape of the original array
    pub shape: Vec<usize>,
    /// Path to folder containing chunks
    path: PathBuf,

    ///.
    _t: PhantomData<T>,
    _d: PhantomData<D>,
}

impl<T: ReadableElement + WritableElement + Clone + Default + Debug, D: Dimension> ChunkedArray<T, D> {
    /// Returns a given slice from the ChunkedArray
    #[must_use]
    pub fn get<Dout: Dimension, const N: usize>(
        &self,
        idxs: SliceInfo<[SliceInfoElem; N], D, Dout>,
    ) -> Array<T, Dout> {
        let idxs = idxs.as_ref();

        // Now we need to resolve which chunks we actually need
        let info = idxs[self.chunked_along];
        let slice_axis_indices: Vec<usize> =
            slice_info_to_vec(&info, self.shape[self.chunked_along]);

        // Before we load in the chunks we first need to calculate which chunks we'll even need
        let start_chunk = slice_axis_indices[0] / self.chunk_size;
        let end_chunk = slice_axis_indices[slice_axis_indices.len()] / self.chunk_size;

        // Now load in the actual chunks
        let chunks: Vec<Array<T, D>> = (start_chunk..end_chunk)
            .map(|n| {
                let path = self.path.join(format!("chunk{n}.npy"));
                read_npy(path).unwrap()
            })
            .collect();

        // Now we need to iterate along the sliced axis and slice correctly
        //Todo this, we need to generate a new list of SliceInfoElems
        
        for n in slice_axis_indices {
            let n = n - start_chunk;
            idxs[self.chunked_along] = SliceInfoElem::Index(n as isize);

            let chunk_needed = 0;
            chunks[chunk_needed].slice(idxs);
        }

        // Then, for each index we actually need, we want to change the slice

        Default::default()
    }

    ///.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        let total_along_axis = self.shape[self.chunked_along];
        (total_along_axis / self.chunk_size) + (total_along_axis % self.chunk_size)
    }

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
            _d: PhantomData,
        })
    }

    ///.
    /// # Errors
    fn load_metadata(folder: &Path) -> Result<(Vec<usize>, usize, usize), String> {
        // Everything is u32
        //ndim; shape; axis along which the array was split; the max size of each chunk
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

    /// .
    /// # Errors
    pub fn chunk(
        arr: &Array<T, D>,
        settings: &ChunkSettings,
    ) -> Result<(), String> {
        let path = Path::new(&settings.path);

        // Create the directory, etc.
        Self::handle_directory(path, settings.delete_dir)?;

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

    #[allow(clippy::cast_possible_truncation)]
    /// .
    fn generate_metadata<A: WritableElement>(
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

    /// Creates the new directory (if necesary)
    fn handle_directory(path: &Path, delete_dir: bool) -> Result<(), String> {
        if path.exists() {
            if !path.is_dir() {
                return Err("Path exists and is not a directory".to_string());
            }

            // At this point we know its a directory so we can just check if we're allowed to delete it
            if !delete_dir && !directory_is_empty(path)? {
                return Err("Directory exists and `delete_dir` is set to false".to_string());
            }

            // If we're here, then we can delete the directory and its contents
            // TODO
        } else {
            // If we're here then we can create the directory
            create_dir(path).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}
