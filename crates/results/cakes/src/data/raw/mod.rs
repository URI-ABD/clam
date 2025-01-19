//! Reading data from various sources.

use abd_clam::{dataset::AssociatesMetadataMut, FlatVec};

use super::tree::instances::{Aligned, MemberSet, Unaligned};

mod ann_benchmarks;
pub mod fasta;

/// The datasets we use for benchmarks.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types, clippy::doc_markdown, clippy::module_name_repetitions)]
#[non_exhaustive]
pub enum RawData {
    /// The GreenGenes 12.10 dataset.
    #[clap(name = "gg_12_10")]
    GreenGenes_12_10,
    /// The pre-aligned GreenGenes 12.10 dataset.
    #[clap(name = "gg_aligned_12_10")]
    GreenGenesAligned_12_10,
    /// The GreenGenes 13.5 dataset.
    #[clap(name = "gg_13_5")]
    GreenGenes_13_5,
    /// The Silva 18S dataset.
    #[clap(name = "silva_18S")]
    Silva_18S,
    /// The pre-aligned Silva 18S dataset.
    #[clap(name = "silva_aligned_18S")]
    SilvaAligned_18S,
    /// The PDB sequence dataset.
    #[clap(name = "pdb_seq")]
    PdbSeq,
    /// The Kosarak dataset.
    #[clap(name = "kosarak")]
    Kosarak,
    /// The MovieLens-10M dataset.
    #[clap(name = "movielens_10m")]
    MovieLens_10M,
}

impl RawData {
    /// Returns the name of the dataset as a string.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::GreenGenes_12_10 => "gg_12_10",
            Self::GreenGenesAligned_12_10 => "gg_aligned_12_10",
            Self::GreenGenes_13_5 => "gg_13_5",
            Self::Silva_18S => "silva_18S",
            Self::SilvaAligned_18S => "silva_aligned_18S",
            Self::PdbSeq => "pdb_seq",
            Self::Kosarak => "kosarak",
            Self::MovieLens_10M => "movielens_10m",
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
    pub fn read<P: AsRef<std::path::Path>>(self, inp_path: &P, out_dir: &P) -> Result<super::tree::Tree, String> {
        let out_dir = out_dir.as_ref();
        let (data_path, queries_path, gt_path) = {
            let mut data_path = out_dir.to_path_buf();
            data_path.push(self.data_name());

            let mut queries_path = out_dir.to_path_buf();
            queries_path.push(self.queries_name());

            let mut gt_path = out_dir.to_path_buf();
            gt_path.push(self.gt_name());

            (data_path, queries_path, gt_path)
        };

        match self {
            Self::GreenGenes_12_10 | Self::GreenGenes_13_5 | Self::Silva_18S | Self::PdbSeq => {
                let (data, queries) = if data_path.exists() && queries_path.exists() {
                    ftlog::info!("Reading data from {data_path:?}");
                    let bytes: Vec<u8> = std::fs::read(&data_path).map_err(|e| e.to_string())?;
                    let data = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Reading queries from {queries_path:?}");
                    let bytes: Vec<u8> = std::fs::read(&queries_path).map_err(|e| e.to_string())?;
                    let queries = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

                    (data, queries)
                } else {
                    let ([data, queries], [min_len, max_len]) = fasta::read(inp_path, 1000, false)?;
                    let (metadata, data): (Vec<_>, Vec<_>) =
                        data.into_iter().map(|(name, seq)| (name, Unaligned::from(seq))).unzip();

                    let data = FlatVec::new(data)?
                        .with_metadata(&metadata)?
                        .with_dim_lower_bound(min_len)
                        .with_dim_upper_bound(max_len);

                    let queries = queries
                        .into_iter()
                        .map(|(name, seq)| (name, Unaligned::from(seq)))
                        .collect();

                    ftlog::info!("Writing data to {data_path:?}");
                    let bytes = bitcode::encode(&data).map_err(|e| e.to_string())?;
                    std::fs::write(&data_path, &bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Writing queries to {queries_path:?}");
                    let bytes = bitcode::encode(&queries).map_err(|e| e.to_string())?;
                    std::fs::write(&queries_path, &bytes).map_err(|e| e.to_string())?;

                    (data, queries)
                };

                super::tree::Tree::new_unaligned(self.name(), out_dir, data, queries)
            }
            Self::GreenGenesAligned_12_10 | Self::SilvaAligned_18S => {
                let (data, queries) = if data_path.exists() && queries_path.exists() {
                    ftlog::info!("Reading data from {data_path:?}");
                    let bytes: Vec<u8> = std::fs::read(&data_path).map_err(|e| e.to_string())?;
                    let data = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Reading queries from {queries_path:?}");
                    let bytes: Vec<u8> = std::fs::read(&queries_path).map_err(|e| e.to_string())?;
                    let queries = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

                    (data, queries)
                } else {
                    let ([data, queries], [min_len, max_len]) = fasta::read(inp_path, 1000, false)?;
                    let (metadata, data): (Vec<_>, Vec<_>) =
                        data.into_iter().map(|(name, seq)| (name, Aligned::from(seq))).unzip();

                    let data = FlatVec::new(data)?
                        .with_metadata(&metadata)?
                        .with_dim_lower_bound(min_len)
                        .with_dim_upper_bound(max_len);

                    let queries = queries
                        .into_iter()
                        .map(|(name, seq)| (name, Aligned::from(seq)))
                        .collect();

                    ftlog::info!("Writing data to {data_path:?}");
                    let bytes = bitcode::encode(&data).map_err(|e| e.to_string())?;
                    std::fs::write(&data_path, &bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Writing queries to {queries_path:?}");
                    let bytes = bitcode::encode(&queries).map_err(|e| e.to_string())?;
                    std::fs::write(&queries_path, &bytes).map_err(|e| e.to_string())?;

                    (data, queries)
                };

                super::tree::Tree::new_aligned(self.name(), out_dir, data, queries)
            }
            Self::Kosarak | Self::MovieLens_10M => {
                let (data, queries, ground_truth) = if data_path.exists() && queries_path.exists() && gt_path.exists() {
                    ftlog::info!("Reading data from {data_path:?}");
                    let bytes = std::fs::read(&data_path).map_err(|e| e.to_string())?;
                    let data = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Reading queries from {queries_path:?}");
                    let bytes = std::fs::read(&queries_path).map_err(|e| e.to_string())?;
                    let queries = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Reading ground truth from {gt_path:?}");
                    let bytes = std::fs::read(&gt_path).map_err(|e| e.to_string())?;
                    let ground_truth = bitcode::decode(&bytes).map_err(|e| e.to_string())?;

                    (data, queries, ground_truth)
                } else {
                    let data = ann_benchmarks::read::<_, usize>(inp_path, true)?;
                    let (data, queries, ground_truth) = (data.train, data.queries, data.neighbors);

                    let data = data.iter().map(MemberSet::<_>::from).collect();
                    let data = FlatVec::new(data)?;

                    let queries = queries.iter().map(MemberSet::<_>::from).enumerate().collect();

                    ftlog::info!("Writing data to {data_path:?}");
                    let bytes = bitcode::encode(&data).map_err(|e| e.to_string())?;
                    std::fs::write(&data_path, &bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Writing queries to {queries_path:?}");
                    let bytes = bitcode::encode(&queries).map_err(|e| e.to_string())?;
                    std::fs::write(&queries_path, &bytes).map_err(|e| e.to_string())?;

                    ftlog::info!("Writing ground truth to {gt_path:?}");
                    let bytes = bitcode::encode(&ground_truth).map_err(|e| e.to_string())?;
                    std::fs::write(&gt_path, &bytes).map_err(|e| e.to_string())?;

                    (data, queries, ground_truth)
                };

                super::tree::Tree::new_ann_set(self.name(), out_dir, data, queries, ground_truth)
            }
        }
    }

    /// Returns the name of the file containing the uncompressed data as a serialized `FlatVec`.
    fn data_name(&self) -> String {
        format!("{}.flat_data", self.name())
    }

    /// Returns the name of the file containing the queries as a serialized `Vec`.
    fn queries_name(&self) -> String {
        format!("{}.queries", self.name())
    }

    /// Returns the name of the file containing the ground truth as a serialized `Vec`.
    fn gt_name(&self) -> String {
        format!("{}.ground_truth", self.name())
    }
}
