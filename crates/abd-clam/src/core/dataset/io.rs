//! Traits for Disk IO operations with datasets.

use super::{Dataset, ParDataset};

#[cfg(feature = "disk-io")]
/// For writing and reading datasets to and from disk.
pub trait DatasetIO<I>: Dataset<I> + bitcode::Encode + bitcode::Decode {
    /// Writes the `Dataset` to disk in binary format using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If the dataset cannot be encoded.
    /// - If the file cannot be written.
    fn write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let bytes = bitcode::encode(self).map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    /// Reads the `Dataset` from disk in binary format using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If the file cannot be read.
    /// - If the dataset cannot be decoded.
    fn read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&bytes).map_err(|e| e.to_string())
    }
}

#[cfg(feature = "disk-io")]
/// Parallel version of [`DatasetIO`](crate::core::dataset::io::DatasetIO).
pub trait ParDatasetIO<I: Send + Sync>: DatasetIO<I> + ParDataset<I> {
    /// Parallel version of [`DatasetIO::write_to`](crate::core::dataset::io::DatasetIO::write_to).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`DatasetIO::write_to`](crate::core::dataset::io::DatasetIO::write_to).
    fn par_write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        self.write_to(path)
    }

    /// Parallel version of [`DatasetIO::read_from`](crate::core::dataset::io::DatasetIO::read_from).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`DatasetIO::read_from`](crate::core::dataset::io::DatasetIO::read_from).
    fn par_read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        Self::read_from(path)
    }
}
