use super::_constructable::ConstructableNumber;
/// The `BatchedArrowReader` is the file interface this library uses to deal with
/// the Arrow IPC format and batched data.
use super::{
    io::{process_data_directory, read_bytes_from_file},
    metadata::ArrowMetaData,
};
use arrow_format::ipc::Buffer;
use std::path::PathBuf;
use std::{error::Error, marker::PhantomData};
use std::{fs::File, sync::RwLock};

/// Encapsulates the original and reordered indices that a dataset may have
#[derive(Debug)]
pub struct ArrowIndices {
    /// The original ordering of the dataset (may or may not be linear)
    pub original_indices: Vec<usize>,

    /// The reordering map for this dataset
    pub reordered_indices: Vec<usize>,
}

/// Orchestrates and coordinates reading and random access to a set of dataset
/// shards in the Arrow IPC format.
#[derive(Debug)]
pub struct BatchedArrowReader<T: ConstructableNumber> {
    /// The original and reordered indices
    pub indices: ArrowIndices,

    /// The directory where the data is stored
    data_dir: PathBuf,

    /// The metadata associated with this batch of files
    metadata: ArrowMetaData<T>,

    /// The readers for each of the shards in the dataset
    readers: RwLock<Vec<File>>,

    /// The number of readers in the dataset. This is just an alias to
    /// self.readers.len() but allows us to cut down on the number of
    /// reads to the `RwLock`
    num_readers: usize,

    /// A preallocated vector of u8's corresponding to an instance in
    /// the dataset. This is a member to cut down on the number of allocations
    /// needed
    col: RwLock<Vec<u8>>,

    /// We only accept a single primitive type for the dataset. This is that type
    _t: PhantomData<T>,
}

impl<T: ConstructableNumber> BatchedArrowReader<T> {
    /// Constructs a new `BatchedArrowReader`
    ///
    /// # Args
    /// - `data_dir`: The dataset directory
    ///
    /// # Returns
    /// A newly constructed `BatchedArrowReader` situated in `data_dir`
    ///
    /// # Errors
    /// Any I/O or IPC format errors
    pub fn new(data_dir: &str) -> Result<Self, Box<dyn Error>> {
        // By processing our data directory we get both the handles for each of the shards and the reordering map
        // for the dataset if it exists
        let path = PathBuf::from(data_dir);
        let (mut handles, reordered_indices) = process_data_directory(&path)?;

        let num_readers = handles.len();

        // Load in the metadata from the first and last file in the batch
        let metadata = ArrowMetaData::<T>::try_from(&mut handles)?;
        let cardinality = metadata.calculate_cardinality(num_readers);

        // Index information
        let original_indices: Vec<usize> = (0..cardinality).collect();
        let reordered_indices = reordered_indices.map_or_else(|| original_indices.clone(), |indices| indices);

        Ok(Self {
            data_dir: path,
            indices: ArrowIndices {
                original_indices,
                reordered_indices,
            },

            readers: RwLock::new(handles),
            num_readers,
            _t: PhantomData,
            col: RwLock::new(vec![0u8; metadata.row_size_in_bytes()]),
            metadata,
        })
    }

    /// Returns a column at a given index in the dataset
    ///
    /// # Args
    /// - `idx`: The desired index
    ///
    /// # Returns
    /// A vector of scalars representing an instance in the dataset
    ///
    pub fn get(&self, index: usize) -> Vec<T> {
        let index = self.indices.reordered_indices[index];
        let metadata = &self.metadata;

        // Returns the index of the reader associated with the index
        let reader_index: usize = (index - (index % metadata.cardinality_per_batch)) / metadata.cardinality_per_batch;

        // Get the relative index
        let index: usize = index % metadata.cardinality_per_batch;

        // Becuase we're limited to primitive types, we only have to deal with buffer 0 and
        // buffer 1 which are the validity and data buffers respectively. Therefore for every
        // index, there are two buffers associated with that column, the second of which is
        // the data buffer, hence the 2*i+1.
        let data_buffer: Buffer = metadata.buffers[index];

        // Resolve the offset that we need to seek to to get to the actual data. This is
        // conditonally different based upon the last shard in the dataset's cardinality.
        #[allow(clippy::cast_sign_loss)]
        let offset = if reader_index == self.num_readers - 1 {
            metadata.last_batch_start_of_data + data_buffer.offset as u64
        } else {
            metadata.start_of_data + data_buffer.offset as u64
        };

        // We `expect` here because any other result is a total failure
        #[allow(clippy::expect_used)]
        let mut readers = self.readers.write().expect("Could not access column. Invalid index");

        #[allow(clippy::expect_used)]
        let mut col = self.col.write().expect("Could not access column buffer. Memory error.");

        read_bytes_from_file(&mut readers[reader_index], offset, &mut col)
    }

    /// Writes a reordering map to the dataset's data directory
    ///
    /// # Errors
    /// Any I/O or IPC formatting errors that occur during constructing and writing the reordering
    /// map
    pub fn write_reordering_map(&self) -> Result<(), Box<dyn Error>> {
        super::io::write_reordering_map(&self.indices.reordered_indices, &self.data_dir)
    }
}
