//! Traits for Disk IO with structs from the crate.

/// Writes and reads structs to and from disk in binary.
#[cfg(feature = "disk-io")]
pub trait DiskIO: Sized {
    /// Convert the struct to a byte vector in binary.
    ///
    /// # Errors
    ///
    /// Depending on the underlying implementation.
    fn to_bytes(&self) -> Result<Vec<u8>, String>;

    /// Convert a byte vector in binary to the struct.
    ///
    /// # Errors
    ///
    /// Depending on the underlying implementation.
    fn from_bytes(bytes: &[u8]) -> Result<Self, String>;

    /// Writes the struct to disk in binary.
    ///
    /// # Errors
    ///
    /// - If the struct cannot be encoded.
    /// - If the file cannot be written.
    fn write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()?).map_err(|e| e.to_string())
    }

    /// Reads the struct from disk in binary format using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If the file cannot be read.
    /// - If the struct cannot be decoded.
    fn read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        Self::from_bytes(&bytes)
    }
}

/// Parallel version of [`DiskIO`](DiskIO).
#[cfg(feature = "disk-io")]
pub trait ParDiskIO: DiskIO + Send + Sync {
    /// Parallel version of [`DiskIO::to_bytes`](DiskIO::to_bytes).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`DiskIO::to_bytes`](DiskIO::to_bytes).
    fn par_to_bytes(&self) -> Result<Vec<u8>, String> {
        self.to_bytes()
    }

    /// Parallel version of [`DiskIO::from_bytes`](DiskIO::from_bytes).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`DiskIO::from_bytes`](DiskIO::from_bytes).
    fn par_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::from_bytes(bytes)
    }

    /// Parallel version of [`DiskIO::write_to`](DiskIO::write_to).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`BitCodeIO::write_to`](DiskIO::write_to).
    fn par_write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        self.write_to(path)
    }

    /// Parallel version of [`DiskIO::read_from`](DiskIO::read_from).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`DiskIO::read_from`](DiskIO::read_from).
    fn par_read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        Self::read_from(path)
    }
}
