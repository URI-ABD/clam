use arrow2::{
    array::{PrimitiveArray, UInt64Array},
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
    io::ipc::read::{read_file_metadata, FileReader},
    io::ipc::write::{FileWriter, WriteOptions},
};
use std::error::Error;
use std::io::{Read, Seek, SeekFrom};
use std::{
    ffi::OsString,
    fs::{read_dir, DirEntry, File},
    path::{Path, PathBuf},
};

use super::_constructable::ConstructableNumber;

/// The default name for a reordering map
const REORDERING_FILENAME: &str = "reordering.arrow";

/// The result of processing a data directory. Namely, a list of file
/// handles and an optional reordering map if the dataset has already
/// been reordered.
pub type FilesAndReorderingMap = (Vec<File>, Option<Vec<usize>>);

/// Scans a given directory for batch files and returns their handles as well as an optional
/// set of reordered indices which are read from a specific file. If no reordering map is
/// present the function will return just the file handles and `None`.
///
/// ## Note on file read order
/// The files will be read based off of lexicographical order. Meaning if the order if your
/// data is important, the files should be named in such a way that earlier files will sort
/// earlier. The reasoning for this is that `read_dir` has no cross-platform consistent
/// ordering, thus to maintain the validity of the reordering map, it is necessary to
/// standardize the order in which files are read in.
///
/// # Arguments
/// `data_dir`: A directory pointing to a batched dataset
pub fn process_data_directory(data_dir: &Path) -> Result<FilesAndReorderingMap, Box<dyn Error>> {
    let mut reordering = None;
    let mut filenames: Vec<OsString> = read_dir(data_dir)?
        .collect::<Result<Vec<DirEntry>, std::io::Error>>()?
        .into_iter()
        .map(|file| file.file_name())
        .collect();

    filenames.sort();

    if filenames.iter().any(|file| file == REORDERING_FILENAME) {
        reordering = Some(read_reordering_map(data_dir)?);
    }

    let handles: Vec<File> = filenames
        .iter()
        .filter(|name| *name != REORDERING_FILENAME)
        .map(|name| File::open(data_dir.join(name)))
        .collect::<Result<Vec<File>, std::io::Error>>()?;

    Ok((handles, reordering))
}

/// Writes a set of indices to a new arrow file located in `data_dir`
///
/// # Args
/// - `reordered_indices`: A reordering map for a given dataset
/// - `data_dir`: The directory to place the reordering map
pub fn write_reordering_map(reordered_indices: &[usize], data_dir: &Path) -> Result<(), Box<dyn Error>> {
    let reordered_indices: Vec<u64> = reordered_indices.iter().map(|x| *x as u64).collect();

    let array: PrimitiveArray<u64> = UInt64Array::from_vec(reordered_indices);

    let schema = Schema::from(vec![Field::new("Reordering", DataType::UInt64, true)]);

    let file = File::create(data_dir.join(REORDERING_FILENAME))?;
    let options = WriteOptions { compression: None };
    let mut writer = FileWriter::try_new(file, schema, None, options)?;
    let chunk = Chunk::try_new(vec![array.boxed()])?;

    writer.write(&chunk, None)?;
    writer.finish()?;

    Ok(())
}

/// Reads a number of bytes from reader starting at position `offset`. This function is primarily
/// used to read a number of bytes at some known position in a file. The `reader.seek` call gets
/// compiled down to lseek(1) on linux which is constant time, so this function is bounded in
/// complexity linearly with respect to the size of the buffer.
///
/// # Note
/// This function will panic if either the seek position is invalid (out of bounds) or the bufffer
/// cannot be filled. Either of these states are invalid and thus the panic is justified.
///
/// # Args
/// - `reader`: A file
/// - `offset`: The number of bytes from the start of the file we should start reading after. I.e.
///     if offset is `n`, this function will begin reading at position `n` in the file.
#[allow(clippy::unwrap_used)]
pub fn read_bytes_from_file<T: ConstructableNumber>(reader: &mut File, offset: u64, buffer: &mut [u8]) -> Vec<T> {
    // Here's where we do the mutating
    // Skip past the validity bytes (our data is assumed to be non-nullable)
    reader
        .seek(SeekFrom::Start(
            // The data buffer's offset is the start of the actual data.
            offset,
        ))
        .unwrap();

    // We then load the data of this row into the column data buffer
    reader.read_exact(buffer).unwrap();

    // Map the bytes to our type
    buffer
        .chunks(std::mem::size_of::<T>())
        .map(|chunk| T::from_ne_bytes(chunk))
        .collect()
}

/// Reads in a reordering map in a diretory and returns the reordering
///
/// # Note
/// At the moment this function panics if the file does not contain any columns
///
/// # Args
/// - `data_dir`: The directory where the reordering map is located
fn read_reordering_map(data_dir: &Path) -> Result<Vec<usize>, Box<dyn Error>> {
    // Load in the file
    let mut reader = File::open(data_dir.join(PathBuf::from(REORDERING_FILENAME)))?;

    // Load in its metadata using arrow2
    let metadata = read_file_metadata(&mut reader)?;
    let mut reader = FileReader::new(reader, metadata, None, None);

    // There's only one column, so we grab it
    let binding = reader
        .next()
        .ok_or("Could not read first column. Reordering map is invalid.")??;
    let column = &binding.columns()[0];

    // `Array` implements `Any`, so we can downcase it to a PrimitiveArray<u64> without any isssues, then
    // just convert that to usize. Unwrapping here is fine because we assume non-nullable because
    // `write_reordering_map` writes as non-nullable.
    column
        .as_any()
        .downcast_ref::<PrimitiveArray<u64>>()
        .ok_or("Could not read reordering map. Invalid file")?
        .into_iter()
        .map(convert_u64_usize)
        .collect()
}

/// Converts a u64 to a usize with appropriate error handling
///
/// # Args
/// - `num`: Possibly a u64
///
/// # Returns
/// A result with the converted usize
///
/// # Errors
/// If the argument is empty, this constitutes an error. There may also
/// be conversion errors from u64 to usize
///
/// # Note
/// Does this affect embedded applications?
fn convert_u64_usize(num: Option<&u64>) -> Result<usize, Box<dyn Error>> {
    let ref_num = num.ok_or("Missing index in reordering map. Invalid file.")?;
    let converted = usize::try_from(*ref_num)?;
    Ok(converted)
}
