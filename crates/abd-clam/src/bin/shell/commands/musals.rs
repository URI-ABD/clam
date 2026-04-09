//! Commands for the Musals CLI.

use std::{collections::HashSet, path::PathBuf};

use abd_clam::{common_metrics, musals::MeasurableAlignmentQuality};
use clap::Subcommand;

use crate::{
    tree::{ShellTree, levenshtein::LevenshteinTree},
    utils::ReportFormat,
};

/// The specific action to perform with Musals.
#[derive(Subcommand, Debug)]
#[non_exhaustive]
pub enum Action {
    /// Create an MSA using the given cost matrix.
    Align {
        /// The cost matrix to use for the alignment. If not provided, will use the `Simple` cost matrix.
        #[arg(short('c'), long)]
        cost_matrix: Option<ShellCostMatrix>,
    },
    /// Evaluate the quality of an MSA using the given quality metrics.
    Evaluate {
        /// The name of the quality metrics to use for evaluation. If not provided, will use all available metrics.
        #[arg(short('m'), long)]
        quality_metrics: Option<Vec<MeasurableAlignmentQuality>>,
        /// The number of sequences to sample from the MSA for evaluation. If not provided, will use 1000 sequences.
        #[arg(short('n'), long)]
        num_samples: Option<usize>,
        /// The path to the reference MSA to compare against. If not provided, will cause errors if the metric requires a reference MSA.
        #[arg(short('r'), long)]
        reference_msa: Option<PathBuf>,
    },
}

/// The different cost matrices available for use in the Align action.
#[derive(Debug, Clone, Copy, clap::ValueEnum, Default)]
pub enum ShellCostMatrix {
    /// A simple cost matrix for nucleotide and protein sequences, where matches have a cost of 0, mismatches have a cost of 1, and gaps have a cost of 1.
    #[clap(name = "simple")]
    #[default]
    Simple,
    /// The same as the Simple cost matrix, but with a gap opening cost of 10 and a gap extension cost of 1.
    #[clap(name = "affine")]
    Affine,
    /// The extended-IUPAC cost matrix for nucleotide sequences using the extended IUPAC codes.
    #[clap(name = "extended-iupac")]
    ExtendedIupac,
    /// The BLOSUM62 cost matrix for protein sequences.
    #[clap(name = "blosum62")]
    Blosum62,
}

impl core::fmt::Display for ShellCostMatrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Affine => write!(f, "affine"),
            Self::ExtendedIupac => write!(f, "extended-iupac"),
            Self::Blosum62 => write!(f, "blosum62"),
        }
    }
}

impl Action {
    /// Perform the specified action on the given MSA file, and write the output to the given output directory in the specified output format.
    pub fn perform<P: AsRef<std::path::Path>, R: rand::Rng>(
        &self,
        tree_path: &P,
        out_dir: &std::path::Path,
        out_fmt: ReportFormat,
        rng: &mut R,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Align { cost_matrix } => {
                let cost_matrix = cost_matrix.unwrap_or_default();
                ftlog::info!("Using cost matrix: {cost_matrix}");
                let cost_matrix_name = format!("{cost_matrix}");

                let tree_name = if let Some(name) = tree_path.as_ref().file_name() {
                    name.to_string_lossy().to_string()
                } else {
                    return Err("Input tree path must have a file name.".into());
                };
                let out_path = out_dir.join(format!("aligned-{cost_matrix_name}-{tree_name}"));
                if out_path.exists() {
                    ftlog::warn!("Output aligned tree file already exists: {out_path:?}. It will be overwritten.");
                }

                ftlog::info!("Reading tree of unaligned sequences from {:?}...", tree_path.as_ref());
                let ShellTree::Levenshtein(LevenshteinTree::String(tree)) = ShellTree::read(tree_path)? else {
                    return Err("Input tree must be a Levenshtein tree of strings for the Align action.".into());
                };

                ftlog::info!("Aligning tree using Musals...");
                let cost_matrix = cost_matrix.to_musals_cost_matrix();
                let metric = common_metrics::levenshtein_aligned;
                let tree = ShellTree::Levenshtein(LevenshteinTree::Aligned(tree.par_align(cost_matrix, metric)));

                ftlog::info!("Writing aligned tree to {out_path:?}...");
                tree.write(&out_path)
            }
            Self::Evaluate {
                quality_metrics,
                num_samples,
                reference_msa,
            } => {
                if reference_msa.is_some() {
                    todo!("Comparison against reference MSA not implemented yet.");
                }

                let tree_name = if let Some(name) = tree_path.as_ref().file_name() {
                    name.to_string_lossy().to_string()
                } else {
                    return Err("Input tree path must have a file name.".into());
                };
                let out_path = out_dir.join(format!("quality-{tree_name}.{}", out_fmt.file_extension()));
                if out_path.exists() {
                    ftlog::warn!("Output quality report file already exists: {out_path:?}. It will be overwritten.");
                }

                let quality_metrics = quality_metrics
                    .as_ref()
                    .map_or_else(MeasurableAlignmentQuality::all_variants, |quality_metrics| {
                        quality_metrics.iter().copied().collect::<HashSet<_>>().into_iter().collect()
                    });

                ftlog::info!("Reading tree of aligned sequences from {:?}...", tree_path.as_ref());
                let ShellTree::Levenshtein(LevenshteinTree::Aligned(tree)) = ShellTree::read(tree_path)? else {
                    return Err("Input tree must be a Levenshtein tree of aligned sequences for the Evaluate action.".into());
                };

                let sample_size = num_samples.unwrap_or(1000);
                ftlog::info!("Measuring alignment quality using metrics: {quality_metrics:?} with sample size: {sample_size}...");

                let measurements = quality_metrics
                    .iter()
                    .inspect(|qm| ftlog::info!("Measuring alignment quality using metric: {qm:?}..."))
                    .map(|qm| qm.par_measure_tree(&tree, sample_size, rng))
                    .inspect(|qm| ftlog::info!("Measured alignment quality {qm}"))
                    .collect::<Vec<_>>();

                ftlog::info!("Writing alignment quality report to {out_path:?}...");
                out_fmt.write_report(&measurements, &out_path, true)
            }
        }
    }
}

impl ShellCostMatrix {
    /// Convert the shell cost matrix to a Musals cost matrix.
    pub fn to_musals_cost_matrix(self) -> abd_clam::musals::CostMatrix<usize> {
        match self {
            Self::Simple => abd_clam::musals::CostMatrix::default(),
            Self::Affine => abd_clam::musals::CostMatrix::default_affine(Some(10)),
            Self::ExtendedIupac => abd_clam::musals::CostMatrix::extended_iupac(Some(10)),
            Self::Blosum62 => abd_clam::musals::CostMatrix::blosum62(Some(10)),
        }
    }
}
