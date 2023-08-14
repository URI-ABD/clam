use super::_constructable::ConstructableNumber;
use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::Buffer;
use arrow_format::ipc::MessageHeaderRef::RecordBatch;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::{fmt, mem};

/// The number of bytes every arrow file has preprended by default.
/// These are always skipped and seeked past.
const ARROW_MAGIC_OFFSET: u64 = 12;

/// An error that may occur during metadata parsing
#[derive(Debug)]
pub struct MetadataParsingError<'msg>(&'msg str);

impl<'msg> fmt::Display for MetadataParsingError<'msg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error parsing metadata: {}", self.0)
    }
}

impl<'msg> Error for MetadataParsingError<'msg> {}

/// Metadata for a batch of Arrow IPC files. Specifically, the relevant information
/// from the first and last batches. Leveraging the restrictions placed on datasets
/// laid out in `ArrowDataset`, we can assume all the information we need from the
/// relevant metadata in these two files.
///
/// # Fields
/// - `buffers`: The set of buffer information (offsets where data starts) for
/// the validation and data blocks
/// - `start_of_data`: For every batch except the last, this number is the file
/// position of the beginning of the actual typed data.
/// - `cardinality_per_batch`: The cardinality of every batch except the last
/// - `num_rows`: The number of rows in every batch
/// - `type_size`: The size of the associated type `T`
/// - `last_batch_start_of_data`: The start of the data in the last batch
/// - `last_batch_cardinality`: The cardinality of the last batch
/// (<= `cardinality_per_batch`)
#[derive(Debug)]
pub struct ArrowMetaData<T: ConstructableNumber> {
    /// The offsets of the buffers containing the validation data and actual data
    pub buffers: Vec<Buffer>,

    /// The file pointer offset corresponding to the beginning of the actual data
    pub start_of_data: u64,

    /// The number of instances per batch. Guaranteed for all except the last batch.
    pub cardinality_per_batch: usize,

    /// Number of rows in the dataset (we assume each col. has the same number)
    pub num_rows: usize,

    /// The size of the type of the dataset in bytes
    pub type_size: usize,

    /// The start of the data in the last batch. May or may not be equal to
    /// `start_of_data`
    pub last_batch_start_of_data: u64,

    /// The cardinality of the last batch. May or may not be equal to
    /// `cardinality_per_batch`
    pub last_batch_cardinality: usize,

    /// The primitive type associated with the dataset
    _t: PhantomData<T>,
}

/// Ipc Metadata information.
///
/// In order: Buffers, start of message pointer index, number of rows in the batch
/// (dimensionality), cardinality of the batch
type MetaInfo = (Vec<Buffer>, u64, usize, usize);

impl<T: ConstructableNumber> ArrowMetaData<T> {
    /// Returns the size of a row in the dataset in bytes
    pub const fn row_size_in_bytes(&self) -> usize {
        self.num_rows * self.type_size
    }

    /// Calculates the cardinality of the dataset
    ///
    /// # Args
    /// - `num_readers`: The number of readers (files) in the dataset
    pub const fn calculate_cardinality(&self, num_readers: usize) -> usize {
        self.cardinality_per_batch * (num_readers - 1) + self.last_batch_cardinality
    }

    /// Attempts to construct an `ArrowMetaData` from a given set of handles from the first and last file
    pub fn try_from(handles: &mut [File]) -> Result<Self, Box<dyn Error>> {
        let (buffers, start_of_data, num_rows, cardinality_per_batch) = Self::extract_metadata(&mut handles[0])?;
        let (_, last_batch_start_of_data, _, last_batch_cardinality) =
            Self::extract_metadata(&mut handles[handles.len() - 1])?;

        Ok(Self {
            buffers,
            start_of_data,
            cardinality_per_batch,
            num_rows,
            type_size: mem::size_of::<T>(),
            last_batch_start_of_data,
            last_batch_cardinality,
            _t: PhantomData,
        })
    }

    /// Convenience function which sets a file pointer to the beginning
    /// of the actual data we're interested in
    fn setup_reader(reader: &mut File) -> Result<(), Box<dyn Error>> {
        reader
            .seek(SeekFrom::Start(ARROW_MAGIC_OFFSET))
            .map_err(|_| MetadataParsingError("Could not seek to start of metadata"))?;

        Ok(())
    }

    /// Reads four bytes from a given reader and converts it to a u32
    fn read_metadata_size(reader: &mut File) -> Result<u32, Box<dyn Error>> {
        let mut four_byte_buf: [u8; 4] = [0u8; 4];
        reader
            .read_exact(&mut four_byte_buf)
            .map_err(|_| MetadataParsingError("Could not read metadata size"))?;

        Ok(u32::from_ne_bytes(four_byte_buf))
    }

    /// Attempts to extract IPC metadata from a given file. Note that this function is not
    /// extracting *the* metadata from the file, it's extracting, based on our homogeneity
    /// assumptions, abbreviated information about the first member of the batch from which
    /// we can derive the rest.
    ///
    /// WARNING: Low level, format specific code lies here. <!> BEWARE </!>
    fn extract_metadata(reader: &mut File) -> Result<MetaInfo, Box<dyn Error>> {
        // Setting up the reader means getting the file pointer to the correct position
        Self::setup_reader(reader)?;

        // We then read the next four bytes, this contains a u32 which has the size of the
        // metadata
        let meta_size = Self::read_metadata_size(reader)?;
        let mut data_start = ARROW_MAGIC_OFFSET + u64::from(meta_size);

        // Stuff is always padded to an 8 byte boundary, so we add the padding to the offset
        // The +4 here is to skip past the continuation bytes ff ff ff ff
        data_start += (data_start % 8) + 4;

        // Seek to the start of the actual data.
        // https://arrow.apache.org/docs/format/Columnar.html#encapsulated-message-format
        reader
            .seek(SeekFrom::Start(data_start))
            .map_err(|_| MetadataParsingError("Could not seek to start of data"))?;

        // Similarly, the size of the metadata for the block is also a u32, so we'll read it
        let block_meta_size = Self::read_metadata_size(reader)?;

        // We then actually parse the metadata for the block using flatbuffer. This gives us
        // many things but most notably is the offsets necessary for getting to a given column in
        // a file, as well as the number of rows each column has. This together allows us to read
        // a file.
        let mut meta_buf = vec![0u8; block_meta_size as usize];
        reader
            .read_exact(&mut meta_buf)
            .map_err(|_| MetadataParsingError("Could not fill metadata buffer. Metadata size incorrect."))?;

        let message = arrow_format::ipc::MessageRef::read_as_root(meta_buf.as_ref())
            .map_err(|_| MetadataParsingError("Could not read message. Invalid data."))?;

        // Here we grab the nodes and buffers. Nodes = Row information, basically, and buffers are
        // explained here https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout
        // In short, a buffer is offsets corresponding to a "piece of information". Be it the validity
        // information or the actual data itself.
        //
        // Here we extract the header and the recordbatch that is contained within it. This recordbatch has
        // all of the offset and row/column information we need to traverse the file and get arbitrary access.
        //
        // NOTE: We don't handle anything other than recordbatch headers at the moment.
        //
        // Most of this stuff here comes from the arrow_format crate. We're just extracting the information
        // from the flatbuffer we expect to be in the file.
        let message_header = message
            .header()?
            .ok_or(MetadataParsingError("Message contains no relevant header information"))?;

        // Header is of type MessageHeaderRef, which has a few variants. The only relevant (and valid) one
        // for us is the RecordBatch variant. Therefore, we reject all other constructions at the moment.
        let r = ({
            if let RecordBatch(r) = message_header {
                Ok(r)
            } else {
                Err(MetadataParsingError("Header does not contain record batch"))
            }
        })?;

        // Nodes correspond to, in our case, row information for each column. Therefore nodes.len() is the number
        // of columns in the recordbatch and nodes[0].length() is the number of rows each column has (we assume
        // homogeneous column heights)
        let nodes = r.nodes()?.ok_or(MetadataParsingError(
            "Header contains no node information and thus cannot be read",
        ))?;
        let cardinality_per_batch: usize = nodes.len();
        let num_rows: usize = nodes
            .get(0)
            .ok_or(MetadataParsingError("Header contains no nodes and thus cannot be read"))?
            .length()
            .try_into()?;

        // We then convert the buffer references to owned buffers. This gives us the offset corresponding to the
        // start of each column and the length of each column in bytes.

        // Note: The reason we do `step_by(2)` here is so skip the information related to validation bits.
        let buffers: Vec<Buffer> = r
            .buffers()?
            .ok_or(MetadataParsingError(
                "Metadata contains no buffers and thus cannot be read",
            ))?
            .iter()
            .step_by(2)
            .map(|b| Buffer {
                offset: b.offset(),
                length: b.length(),
            })
            .collect();

        assert_eq!(buffers.len(), cardinality_per_batch);

        // We then grab the start position of the message. This allows us to calculate our offsets
        // correctly. All of the offsets in the buffers are relative to this point.
        let start_of_data: u64 = reader
            .stream_position()
            .map_err(|_| MetadataParsingError("Could not reset file cursor to beginning of file"))?;

        Ok((buffers, start_of_data, num_rows, cardinality_per_batch))
    }
}
