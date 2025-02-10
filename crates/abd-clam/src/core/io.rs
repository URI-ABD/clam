//! Traits for Disk IO with structs from the crate.

/// Writes and reads structs to and from disk in binary format using `bitcode`.
#[cfg(feature = "disk-io")]
pub trait DiskIO: bitcode::Encode + bitcode::Decode {
    /// Writes the struct to disk in binary format using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If the struct cannot be encoded.
    /// - If the file cannot be written.
    fn write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let bytes = bitcode::encode(self).map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    /// Reads the struct from disk in binary format using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If the file cannot be read.
    /// - If the struct cannot be decoded.
    fn read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&bytes).map_err(|e| e.to_string())
    }
}

/// Parallel version of [`DiskIO`](crate::core::io::DiskIO).
#[cfg(feature = "disk-io")]
pub trait ParDiskIO: DiskIO + Send + Sync {
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
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&bytes).map_err(|e| e.to_string())
    }
}
