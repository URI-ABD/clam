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

#[derive(Debug)]
pub(crate) struct ArrowIndices {
    pub original_indices: Vec<usize>,
    pub reordered_indices: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct BatchedArrowReader<T: ConstructableNumber> {
    pub indices: ArrowIndices,

    // The directory where the data is stored
    data_dir: PathBuf,
    metadata: ArrowMetaData<T>,
    readers: RwLock<Vec<File>>,

    // This is here so we dont have to perform two rwlocks every
    // `get`
    num_readers: usize,

    // We allocate a column of the specific number of bytes
    // necessary (type_size * num_rows) at construction to
    // lessen the number of vector allocations we need to do.
    // This might be able to be removed. Unclear.
    _col: RwLock<Vec<u8>>,

    // We'd like to associate this handle with a type, hence the phantomdata
    _t: PhantomData<T>,
}

impl<T: ConstructableNumber> BatchedArrowReader<T> {
    pub(crate) fn new(data_dir: &str) -> Result<Self, Box<dyn Error>> {
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
        let reordered_indices = match reordered_indices {
            Some(indices) => indices,
            None => original_indices.clone(),
        };

        Ok(BatchedArrowReader {
            data_dir: path,
            indices: ArrowIndices {
                reordered_indices,
                original_indices,
            },

            readers: RwLock::new(handles),
            num_readers,
            _t: Default::default(),
            _col: RwLock::new(vec![0u8; metadata.row_size_in_bytes()]),
            metadata,
        })
    }

    pub(crate) fn get(&self, index: usize) -> Vec<T> {
        let resolved_index = self.indices.reordered_indices[index];
        self.get_column(resolved_index)
    }

    fn get_column(&self, index: usize) -> Vec<T> {
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

        let offset = if reader_index == self.num_readers - 1 {
            metadata.last_batch_start_of_data + data_buffer.offset as u64
        } else {
            metadata.start_of_data + data_buffer.offset as u64
        };

        // We `expect` here because any other result is a total failure
        let mut readers = self.readers.write().expect("Could not access column. Invalid index");
        let mut _col = self
            ._col
            .write()
            .expect("Could not access column buffer. Memory error.");

        read_bytes_from_file(&mut readers[reader_index], offset, &mut _col)
    }

    pub(crate) fn write_reordering_map(&self) -> Result<(), Box<dyn Error>> {
        super::io::write_reordering_map(&self.indices.reordered_indices, &self.data_dir)
    }

    pub(crate) fn metadata(&self) -> &ArrowMetaData<T> {
        &self.metadata
    }
}
