//! Reading data from various sources.

use std::fs::File;

use abd_clam::{Dataset, FlatVec, Metric, MetricSpace};

use results_cakes::data::fasta;

/// The datasets we use for benchmarks.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types, clippy::doc_markdown, clippy::module_name_repetitions)]
#[non_exhaustive]
pub enum RawData {
    /// A small hand-crafted dataset.
    #[clap(name = "small")]
    Small,
    /// The GreenGenes 12.10 dataset.
    #[clap(name = "gg_12_10")]
    GreenGenes_12_10,
    /// The GreenGenes 13.5 dataset.
    #[clap(name = "gg_13_5")]
    GreenGenes_13_5,
    /// The Silva 18S dataset.
    #[clap(name = "silva_18S")]
    Silva_18S,
    /// The PDB sequence dataset.
    #[clap(name = "pdb_seq")]
    PdbSeq,
}

impl RawData {
    /// Returns the name of the dataset as a string.
    pub const fn name(&self) -> &str {
        match self {
            Self::Small => "small",
            Self::GreenGenes_12_10 => "gg_12_10",
            Self::GreenGenes_13_5 => "gg_13_5",
            Self::Silva_18S => "silva_18S",
            Self::PdbSeq => "pdb_seq",
        }
    }

    /// Reads the dataset from the given path.
    ///
    /// # Arguments
    ///
    /// * `inp_path`: The path to the file with the raw data.
    /// * `holdout`: The number of queries to hold out. Only used for the fasta datasets.
    /// * `our_dir`: The directory where the output files will be saved.
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
    pub fn read<P: AsRef<std::path::Path>>(
        self,
        inp_path: &P,
        out_dir: &P,
    ) -> Result<FlatVec<String, u32, String>, String> {
        let out_dir = out_dir.as_ref();
        let data_path = {
            let mut data_path = out_dir.to_path_buf();
            data_path.push(self.data_name());
            data_path
        };

        let mut data = if data_path.exists() {
            ftlog::info!("Reading data from {data_path:?}");
            bincode::deserialize_from(File::open(&data_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?
        } else {
            let ([data, _], [min_len, max_len]) = fasta::read(inp_path, 0)?;
            let (metadata, data): (Vec<_>, Vec<_>) = data.into_iter().unzip();

            let data = abd_clam::FlatVec::new(data, Metric::default())?
                .with_metadata(metadata)?
                .with_dim_lower_bound(min_len)
                .with_dim_upper_bound(max_len);

            ftlog::info!("Writing data to {data_path:?}");
            bincode::serialize_into(File::create(&data_path).map_err(|e| e.to_string())?, &data)
                .map_err(|e| e.to_string())?;

            data.with_name(self.name())
        };

        // Set the metric for the data, incase it was deserialized.
        let distance_fn = |x: &String, y: &String| distances::strings::levenshtein::<u32>(x, y);
        let metric = Metric::new(distance_fn, true);
        data.set_metric(metric);

        Ok(data)
    }

    /// Returns the name of the file containing the uncompressed data as a serialized `FlatVec`.
    fn data_name(&self) -> String {
        format!("{}.flat_data", self.name())
    }
}
