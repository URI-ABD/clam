//! Trees built under Levenshtein edit distance.

use std::collections::HashSet;

use abd_clam::{
    Cakes, PartitionStrategy, Tree,
    cakes::{MeasurableSearchQuality, Search},
    common_metrics,
    musals::{AlignedSequence, MusalsTree},
};

use crate::{
    tree::data::{ShellDataType, fasta::read as read_fasta, shuffle_and_truncate},
    utils::SearchBencher,
};

use super::ShellTree;

/// A wrapper because the compiler cannot infer some lifetimes.
#[expect(clippy::ptr_arg)]
fn distance_strings(a: &String, b: &String) -> usize {
    common_metrics::levenshtein_chars(a.chars(), b.chars())
}

/// Trees for Sequence data under Levenshtein distance.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[expect(clippy::type_complexity)]
pub enum LevenshteinTree {
    /// Sequences are represented as `String`s and the distance values are `usize`s.
    String(Tree<String, String, usize, (), fn(&String, &String) -> usize>),
    /// Sequences are represented as [`AlignedSequence`]s and the distance values are `usize`s.
    Aligned(MusalsTree<String, usize, (), fn(&AlignedSequence, &AlignedSequence) -> usize>),
}

impl From<LevenshteinTree> for ShellTree {
    fn from(levenshtein_tree: LevenshteinTree) -> Self {
        Self::Levenshtein(levenshtein_tree)
    }
}

impl databuf::Encode for LevenshteinTree {
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::String(tree) => {
                b"Str".encode::<CONFIG>(buffer)?;
                tree.encode::<CONFIG>(buffer)
            }
            Self::Aligned(tree) => {
                b"Aln".encode::<CONFIG>(buffer)?;
                tree.encode::<CONFIG>(buffer)
            }
        }
    }
}

impl<'de> databuf::Decode<'de> for LevenshteinTree {
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        // Decode the variant name first to determine which variant to decode.
        let variant = <[u8; 3]>::decode::<CONFIG>(buffer)?;
        match &variant {
            b"Str" => {
                let tree = Tree::decode::<CONFIG>(buffer)?;
                Ok(Self::String(tree.with_metric(distance_strings)))
            }
            b"Aln" => {
                let tree: MusalsTree<String, usize, (), ()> = MusalsTree::decode::<CONFIG>(buffer)?;
                Ok(Self::Aligned(tree.with_metric(common_metrics::levenshtein_aligned)))
            }
            _ => Err(format!("Invalid variant for LevenshteinTree: {variant:?}. Expected one of: Str, Aln").into()),
        }
    }
}

impl LevenshteinTree {
    /// Build the tree from the given data and data type.
    ///
    /// # Arguments
    ///
    /// * `data_path` - The path to the input data file.
    /// * `data_type` - The data type to use for building the tree.
    /// * `rng` - The random number generator to use.
    /// * `num_samples` - The number of samples to read from the input data file. If `None`, read all samples.
    /// * `strategy` - The partition strategy to use for building the tree.
    ///
    /// # Returns
    ///
    /// A tuple containing the built tree and a name generated for the output file based on the input data and data type.
    ///
    /// # Errors
    ///
    /// * If the input data file cannot be read or parsed.
    /// * If the data type is not compatible with the Levenshtein metric.
    pub fn build<P: AsRef<std::path::Path>, R: rand::Rng>(
        data_path: &P,
        data_type: &ShellDataType,
        rng: &mut R,
        num_samples: Option<usize>,
        strategy: &PartitionStrategy,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let data = read_fasta(data_path, true)?;
        let data = shuffle_and_truncate(data, rng, num_samples);

        match data_type {
            ShellDataType::String => {
                let metric: fn(&_, &_) -> usize = distance_strings;
                let tree = Tree::par_new(data, metric, &|_| (), &|c| c.cardinality() > 2, strategy)?;
                Ok(Self::String(tree))
            }
            ShellDataType::Aligned => {
                Err("Aligned trees cannot be built directly. Please build a string tree first and then use the `musals` subcommand to create an aligned tree from it.".into())
            }
            _ => Err(format!("Invalid data type for building a tree with the levenshtein metric: {data_type:?}. Expected: 'string'").into()),
        }
    }

    /// Searches the tree using the given algorithm and path to the query data.
    pub fn search<P: AsRef<std::path::Path>>(
        &self,
        algorithm: &str,
        queries_path: &P,
        num_queries: Option<usize>,
    ) -> Result<super::SearchHits, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::String(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<usize>>()?;

                let queries = {
                    let mut queries = read_fasta(queries_path, true)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries.into_iter().map(|(_, seq)| seq).collect::<Vec<_>>()
                };

                let hits = algorithm.par_batch_search(tree, &queries);
                Ok(super::SearchHits::Usize(hits))
            }
            Self::Aligned(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<usize>>()?;

                let queries = {
                    let mut queries = read_fasta(queries_path, true)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries.into_iter().map(|(_, seq)| AlignedSequence::from(seq)).collect::<Vec<_>>()
                };

                let hits = algorithm.par_batch_search(tree, &queries);
                Ok(super::SearchHits::Usize(hits))
            }
        }
    }

    /// Benchmark the algorithm using the given queries and quality metrics.
    pub fn bench<P: AsRef<std::path::Path>>(
        &self,
        algorithm: &str,
        queries_path: &P,
        num_queries: Option<usize>,
        quality_metrics: HashSet<MeasurableSearchQuality>,
    ) -> Result<SearchBencher, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::String(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<usize>>()?;

                let queries = {
                    let mut queries = read_fasta(queries_path, true)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries.into_iter().map(|(_, seq)| seq).collect::<Vec<_>>()
                };

                super::bench_tree(tree, &queries, &algorithm, quality_metrics)
            }
            Self::Aligned(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<usize>>()?;

                let queries = {
                    let mut queries = read_fasta(queries_path, true)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries.into_iter().map(|(_, seq)| AlignedSequence::from(seq)).collect::<Vec<_>>()
                };

                super::bench_tree(tree, &queries, &algorithm, quality_metrics)
            }
        }
    }
}
