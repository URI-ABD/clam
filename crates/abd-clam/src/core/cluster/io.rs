//! Writing `Cluster` trees to CSV files.

use std::io::Write;

use distances::Number;
use rayon::prelude::*;

use super::{Cluster, ParCluster};

#[cfg(feature = "disk-io")]
/// Write a tree to a CSV file.
pub trait Csv<T: Number>: Cluster<T> {
    /// Returns the names of the columns in the CSV file.
    fn header(&self) -> Vec<String>;

    /// Returns a row, corresponding to the `Cluster`, for the CSV file.
    fn row(&self) -> Vec<String>;

    /// Write to a CSV file, all the clusters in the tree.
    ///
    /// # Errors
    ///
    /// - If the file cannot be created.
    /// - If the file cannot be written to.
    /// - If the header cannot be written to the file.
    /// - If any row cannot be written to the file.
    fn write_to_csv<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let line = |items: Vec<String>| {
            let mut line = items.join(",");
            line.push('\n');
            line
        };

        // Create the file and write the header.
        let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        file.write_all(line(self.header()).as_bytes())
            .map_err(|e| e.to_string())?;

        // Write each row to the file.
        for row in self.subtree().into_iter().map(Self::row).map(line) {
            file.write_all(row.as_bytes()).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

#[cfg(feature = "disk-io")]
/// Parallel version of [`Csv`](crate::core::cluster::io::Csv).
pub trait ParCsv<T: Number>: Csv<T> + ParCluster<T> {
    /// Parallel version of [`Csv::write_to_csv`](crate::core::cluster::Csv::write_to_csv).
    ///
    /// # Errors
    ///
    /// See [`Csv::write_to_csv`](crate::core::cluster::Csv::write_to_csv).
    fn par_write_to_csv<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let line = |items: Vec<String>| {
            let mut line = items.join(",");
            line.push('\n');
            line
        };

        // Create the file and write the header.
        let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        file.write_all(line(self.header()).as_bytes())
            .map_err(|e| e.to_string())?;

        let rows = self
            .subtree()
            .into_par_iter()
            .map(Self::row)
            .map(line)
            .collect::<Vec<_>>();

        // Write each row to the file.
        for row in rows {
            file.write_all(row.as_bytes()).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

#[cfg(feature = "disk-io")]
/// Reading and writing `Cluster` trees to disk using `bitcode`.
pub trait ClusterIO<T: Number>: Cluster<T> {
    /// Writes the `Cluster` to disk in binary format using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If the cluster cannot be encoded.
    /// - If the file cannot be written.
    fn write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String>
    where
        Self: bitcode::Encode,
    {
        let bytes = bitcode::encode(self).map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    /// Reads the `Cluster` from disk in binary format using `bitcode`.
    ///
    /// # Errors
    ///
    /// - If the file cannot be read.
    /// - If the cluster cannot be decoded.
    fn read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String>
    where
        Self: bitcode::Decode,
    {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&bytes).map_err(|e| e.to_string())
    }
}

#[cfg(feature = "disk-io")]
/// Parallel version of [`ClusterIO`](crate::core::cluster::io::ClusterIO).
pub trait ParClusterIO<T: Number>: ParCluster<T> + ClusterIO<T> {
    /// Parallel version of [`ClusterIO::write_to`](crate::core::cluster::ClusterIO::write_to).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`ClusterIO::write_to`](crate::core::cluster::ClusterIO::write_to).
    fn par_write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String>
    where
        Self: bitcode::Encode,
    {
        self.write_to(path)
    }

    /// Parallel version of [`ClusterIO::read_from`](crate::core::cluster::ClusterIO::read_from).
    ///
    /// The default implementation offers no parallelism.
    ///
    /// # Errors
    ///
    /// See [`ClusterIO::read_from`](crate::core::cluster::ClusterIO::read_from).
    fn par_read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String>
    where
        Self: bitcode::Decode,
    {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&bytes).map_err(|e| e.to_string())
    }
}
