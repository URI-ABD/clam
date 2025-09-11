//! Generate synthetic datasets for testing and benchmarking.

mod generate;

use clap::Subcommand;

pub use generate::generate_dataset;

#[derive(Subcommand, Debug)]
pub enum GenerateDataAction {
    /// Generate a synthetic dataset with specified parameters.
    Generate {
        /// Number of vectors to generate (m).
        #[arg(short('n'), long)]
        num_vectors: usize,

        /// Dimensionality of each vector (n).
        #[arg(short('d'), long)]
        dimensions: usize,

        /// Base filename for output files (without extension).
        #[arg(short('f'), long)]
        filename: String,

        /// Output format for the dataset.
        #[arg(short('t'), long, value_parser = clap::value_parser!(crate::data::DataType))]
        data_type: crate::data::DataType,

        /// Output format (npy or fasta).
        #[arg(short('F'), long, default_value_t = crate::data::Format::Npy)]
        format: crate::data::Format,

        /// Partition splits as percentages (e.g., "95,5" for 95/5 split).
        /// If not provided, generates a single file with all vectors.
        #[arg(short('p'), long, value_delimiter = ',')]
        partitions: Option<Vec<usize>>,

        /// Minimum value for generated data (for numeric types).
        #[arg(long, default_value = "0.0")]
        min_val: f64,

        /// Maximum value for generated data (for numeric types).
        #[arg(long, default_value = "1.0")]
        max_val: f64,
    },
}
