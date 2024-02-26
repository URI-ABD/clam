use std::{
    collections::{btree_map::Range, HashSet},
    fmt::Debug,
    fs::{create_dir, File},
    io::{Read, Write},
    marker::PhantomData,
    ops::RangeInclusive,
    path::{Path, PathBuf},
    vec,
};

/*
    Need to understand how slicing is going to work.
*/

use ndarray::{concatenate, Array, ArrayD, Axis, Dim, Dimension, IxDyn};
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

///.
pub enum SliceIdx {
    ///.
    Range(std::ops::Range<usize>),
    /// From a to end
    RangeFrom(std::ops::RangeFrom<usize>),
    /// ...
    RangeFull,
    /// To a to =b
    RangeInclusive(std::ops::RangeInclusive<usize>),
    /// From 0 to a
    RangeTo(std::ops::RangeTo<usize>),
    /// From 0 to =a
    RangeToInclusive(std::ops::RangeToInclusive<usize>),
    ///.
    Index(usize),
}

impl SliceIdx {
    ///.
    fn to_range_inclusive(&self, lower: usize, upper: usize) -> RangeInclusive<usize> {
        match &self {
            Self::Index(i) => *i..=*i,
            Self::Range(r) => r.start..=(r.end - 1),
            Self::RangeFrom(r) => r.start..=upper,
            Self::RangeFull => lower..=(upper - 1),
            Self::RangeInclusive(r) => r.clone(),
            Self::RangeTo(r) => lower..=(r.end - 1),
            Self::RangeToInclusive(r) => lower..=r.end,
        }
    }
}

/// .
pub struct ChunkedArray<T: ReadableElement> {
    ///.
    pub chunked_along: usize,
    ///.
    pub chunk_size: usize,
    ///.
    pub shape: Vec<usize>,
    /// Path to folder containing chunks
    _path: PathBuf,

    _t: PhantomData<T>,
}

impl From<usize> for SliceIdx {
    fn from(value: usize) -> Self {
        Self::Index(value)
    }
}

impl From<std::ops::Range<usize>> for SliceIdx {
    fn from(value: std::ops::Range<usize>) -> Self {
        Self::Range(value)
    }
}

impl From<std::ops::RangeFrom<usize>> for SliceIdx {
    fn from(value: std::ops::RangeFrom<usize>) -> Self {
        Self::RangeFrom(value)
    }
}
impl From<std::ops::RangeFull> for SliceIdx {
    fn from(_value: std::ops::RangeFull) -> Self {
        Self::RangeFull
    }
}
impl From<std::ops::RangeInclusive<usize>> for SliceIdx {
    fn from(value: std::ops::RangeInclusive<usize>) -> Self {
        Self::RangeInclusive(value)
    }
}
impl From<std::ops::RangeTo<usize>> for SliceIdx {
    fn from(value: std::ops::RangeTo<usize>) -> Self {
        Self::RangeTo(value)
    }
}
impl From<std::ops::RangeToInclusive<usize>> for SliceIdx {
    fn from(value: std::ops::RangeToInclusive<usize>) -> Self {
        Self::RangeToInclusive(value)
    }
}
impl<T: ReadableElement + Clone + Default + Debug> ChunkedArray<T> {
    ///.
    pub fn get(&self, idxs: &[SliceIdx]) -> Vec<usize> {
        assert!(idxs.len() == self.shape.len());

        let split_slice = &idxs[self.chunked_along];
        let split_length = self.shape[self.chunked_along];
        let chunk_range = split_slice.to_range_inclusive(0, split_length);

        let begin = chunk_range.start() / self.chunk_size;
        let end = chunk_range.end() / self.chunk_size;

        let needed: Vec<_> = (begin..=end).collect();

        println!("{needed:?}");

        let chunks: Vec<Array<T, IxDyn>> = needed
            .iter()
            .map(|ix| {
                read_npy::<PathBuf, Array<T, IxDyn>>(self._path.join(format!("chunk{ix}.npy")))
                    .unwrap()
            })
            .collect();

        let mut end = chunks[0].to_owned();
        for chunk in &chunks[1..] {
            end = concatenate(Axis(self.chunked_along), &[end.view(), chunk.view()])
                .unwrap()
                .to_owned();
        }

        vec![]
    }

    ///.
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
            _path: folder.to_owned(),
            _t: PhantomData,
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
    pub fn chunk<A: WritableElement, D: Dimension>(
        arr: &Array<A, D>,
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
