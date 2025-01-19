//! Reading data from various sources.

use std::path::{Path, PathBuf};

use abd_clam::{
    dataset::{AssociatesMetadataMut, DatasetIO},
    msa::{Aligner, Sequence},
    Dataset, FlatVec,
};
use distances::Number;
use results_cakes::data::fasta;

/// We exclusively use Fasta files for the raw data.
pub struct FastaFile {
    raw_path: PathBuf,
    out_dir: PathBuf,
    name: String,
}

impl FastaFile {
    /// Creates a new `FastaFile` from the given path.
    pub fn new<P: Into<PathBuf>>(raw_path: P, out_dir: Option<P>) -> Result<Self, String> {
        let raw_path: PathBuf = raw_path.into().canonicalize().map_err(|e| e.to_string())?;
        let name = raw_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .ok_or("No file name found")?;

        let out_dir = if let Some(out_dir) = out_dir {
            out_dir.into()
        } else {
            ftlog::info!("No output directory specified. Using default.");
            let mut out_dir = raw_path
                .parent()
                .ok_or("No parent directory of `inp_dir`")?
                .to_path_buf();
            out_dir.push(format!("{name}_results"));
            if !out_dir.exists() {
                std::fs::create_dir(&out_dir).map_err(|e| e.to_string())?;
            }
            out_dir
        }
        .canonicalize()
        .map_err(|e| e.to_string())?;

        if !raw_path.exists() {
            return Err(format!("Path does not exist: {raw_path:?}"));
        }

        if !raw_path.is_file() {
            return Err(format!("Path is not a file: {raw_path:?}"));
        }

        if !out_dir.exists() {
            return Err(format!("Output directory does not exist: {out_dir:?}"));
        }

        if !out_dir.is_dir() {
            return Err(format!("Output directory is not a directory: {out_dir:?}"));
        }

        Ok(Self {
            raw_path,
            out_dir,
            name,
        })
    }

    /// Returns the name of the fasta file without the extension.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the path to the raw fasta file.
    pub fn raw_path(&self) -> &Path {
        &self.raw_path
    }

    /// Returns the path to the output directory.
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// Reads the dataset from the given path.
    ///
    /// # Arguments
    ///
    /// * `num_samples` - The number of samples to read from the dataset. If `None`, all samples are read.
    ///
    /// # Returns
    ///
    /// The dataset and queries, if they were read successfully.
    ///
    /// # Errors
    ///
    /// * If the dataset is not readable.
    /// * If the dataset is not in the expected format.
    #[allow(clippy::too_many_lines)]
    pub fn read<'a, T: Number>(
        &self,
        num_samples: Option<usize>,
        remove_gaps: bool,
        aligner: &'a Aligner<T>,
    ) -> Result<FlatVec<Sequence<'a, T>, String>, String> {
        let data_path = {
            let mut data_path = self.out_dir.clone();
            data_path.push(self.data_name(num_samples));
            data_path
        };

        let mut data = if data_path.exists() {
            ftlog::info!("Reading data from {data_path:?}");
            let data = FlatVec::<String, String>::read_from(&data_path)?;
            let transformer = |s: String| Sequence::new(s, Some(aligner));
            data.transform_items(transformer)
        } else {
            let (data, min_len, max_len) = {
                let ([mut data, _], _) = fasta::read(&self.raw_path, 0, remove_gaps)?;
                if let Some(num_samples) = num_samples {
                    data.truncate(num_samples);
                }
                let (min_len, max_len) = data
                    .iter()
                    .fold((usize::MAX, usize::MIN), |(min_len, max_len), (_, s)| {
                        let len = s.len();
                        (Ord::min(min_len, len), Ord::max(max_len, len))
                    });
                (data, min_len, max_len)
            };

            ftlog::info!("Kept {} sequences with lengths in [{min_len}, {max_len}].", data.len());

            let (metadata, data): (Vec<_>, Vec<_>) = data.into_iter().unzip();

            let data = abd_clam::FlatVec::new(data)?
                .with_metadata(&metadata)?
                .with_dim_lower_bound(min_len)
                .with_dim_upper_bound(max_len)
                .transform_items(|s| Sequence::new(s, Some(aligner)));

            ftlog::info!("Writing data to {data_path:?}");
            let writable_data = data.clone().transform_items(|s| s.seq().to_string());
            writable_data.write_to(&data_path)?;

            data
        };

        let name = num_samples.map_or_else(
            || self.name().to_string(),
            |num_samples| format!("{}-{}", self.name(), num_samples),
        );
        data = data.with_name(&name);

        Ok(data)
    }

    /// Returns the name of the file containing the uncompressed data as a serialized `FlatVec`.
    fn data_name(&self, num_samples: Option<usize>) -> String {
        num_samples.map_or_else(
            || format!("{}.flat_data", self.name()),
            |num_samples| format!("{}-{num_samples}.flat_data", self.name()),
        )
    }
}
