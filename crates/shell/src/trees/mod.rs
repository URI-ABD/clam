//! Tree formats supported in the CLI.

use std::path::{Path, PathBuf};

use abd_clam::{DistanceValue, PartitionStrategy, Tree, cakes::Search, partition_strategy::MaxSplit};
use databuf::{Decode, Encode, config::DEFAULT as DATABUF_DEFAULT};

use crate::{
    commands::cakes::{AlgorithmResult, QueryResult, SearchResults},
    data::{MusalsSequence, OutputFormat, ShellData},
    metrics::{Metric, cosine, euclidean, lcs},
    search::ShellCakes,
};

macro_rules! tree_type {
    ($id:ty, $i:ty, $t:ty) => {
        Tree<$id, $i, $t, (), fn(&$i, &$i) -> $t>
    };
}

pub enum VectorTree {
    F32(tree_type!(usize, Vec<f32>, f32)),
    F64(tree_type!(usize, Vec<f64>, f64)),
    I8(tree_type!(usize, Vec<i8>, f32)),
    I16(tree_type!(usize, Vec<i16>, f32)),
    I32(tree_type!(usize, Vec<i32>, f32)),
    I64(tree_type!(usize, Vec<i64>, f64)),
    U8(tree_type!(usize, Vec<u8>, f32)),
    U16(tree_type!(usize, Vec<u16>, f32)),
    U32(tree_type!(usize, Vec<u32>, f32)),
    U64(tree_type!(usize, Vec<u64>, f64)),
}

#[expect(clippy::type_complexity)]
pub enum ShellTree {
    Lcs(Tree<String, MusalsSequence, u32, (), fn(&MusalsSequence, &MusalsSequence) -> u32>),
    Levenshtein(Tree<String, MusalsSequence, u32, (), Box<dyn Fn(&MusalsSequence, &MusalsSequence) -> u32 + Send + Sync>>),
    Euclidean(VectorTree),
    Cosine(VectorTree),
}

macro_rules! st_new_vector_arm {
    ($items:ident, $metric:ident, $ty:ty, $st_var:ident, $vt_var:ident) => {{
        let strategy = PartitionStrategy::default();
        let items = $items.into_iter().enumerate().collect::<Vec<_>>();
        let metric: fn(&_, &_) -> $ty = $metric::<_, _, _>;
        let tree = Tree::par_new(items, metric, &strategy, &|_| ())?;
        Ok(ShellTree::$st_var(VectorTree::$vt_var(tree)))
    }};
}

impl ShellTree {
    /// Creates a new tree given a dataset and a metric.
    ///
    /// # Arguments
    ///
    /// - `inp_data`: The input data to build the tree from.
    /// - `metric`: The distance metric to use for the tree.
    ///
    /// # Returns
    ///
    /// A new tree and the transformed data.
    ///
    /// # Errors
    ///
    /// - If the dataset and metric are incompatible. The valid combinations
    ///   are:
    ///   - String data with Levenshtein metric.
    ///   - Float or Integer data with Euclidean or Cosine metrics.
    pub fn new(inp_data: ShellData, metric: &Metric) -> Result<ShellTree, String> {
        match metric {
            Metric::Lcs => match inp_data {
                ShellData::String(items) => {
                    let metric: fn(&MusalsSequence, &MusalsSequence) -> u32 = lcs;
                    let strategy = PartitionStrategy::default().with_max_split(MaxSplit::NineTenths);
                    let tree = Tree::par_new(items, metric, &strategy, &|_| ())?;
                    Ok(ShellTree::Lcs(tree))
                }
                _ => Err("LCS metric can only be used with string data".to_string()),
            },
            Metric::Levenshtein => match inp_data {
                ShellData::String(items) => {
                    let sz_device = stringzilla::szs::DeviceScope::default().map_err(|e| e.to_string())?;
                    let sz_engine = stringzilla::szs::LevenshteinDistances::new(&sz_device, 0, 1, 1, 1).map_err(|e| e.to_string())?;
                    let metric = move |a: &MusalsSequence, b: &MusalsSequence| -> u32 {
                        let distances = sz_engine.compute(&sz_device, &[a.as_ref()], &[b.as_ref()]).unwrap();
                        distances[0] as u32
                    };
                    let metric = Box::new(metric) as Box<dyn Fn(&MusalsSequence, &MusalsSequence) -> u32 + Send + Sync>;

                    let strategy = PartitionStrategy::default().with_max_split(MaxSplit::NineTenths);
                    let tree = Tree::par_new(items, metric, &strategy, &|_| ())?;
                    Ok(ShellTree::Levenshtein(tree))
                }
                _ => Err("Levenshtein metric can only be used with string data".to_string()),
            },
            Metric::Euclidean => match inp_data {
                ShellData::String(_) => Err("Euclidean metric cannot be used for string data".to_string()),
                ShellData::F32(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, F32),
                ShellData::F64(items) => st_new_vector_arm!(items, euclidean, f64, Euclidean, F64),
                ShellData::I8(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, I8),
                ShellData::I16(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, I16),
                ShellData::I32(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, I32),
                ShellData::I64(items) => st_new_vector_arm!(items, euclidean, f64, Euclidean, I64),
                ShellData::U8(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, U8),
                ShellData::U16(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, U16),
                ShellData::U32(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, U32),
                ShellData::U64(items) => st_new_vector_arm!(items, euclidean, f64, Euclidean, U64),
            },
            Metric::Cosine => match inp_data {
                ShellData::String(_) => Err("Cosine distance cannot be used for string data".to_string()),
                ShellData::F32(items) => st_new_vector_arm!(items, cosine, f32, Cosine, F32),
                ShellData::F64(items) => st_new_vector_arm!(items, cosine, f64, Cosine, F64),
                ShellData::I8(items) => st_new_vector_arm!(items, cosine, f32, Cosine, I8),
                ShellData::I16(items) => st_new_vector_arm!(items, cosine, f32, Cosine, I16),
                ShellData::I32(items) => st_new_vector_arm!(items, cosine, f32, Cosine, I32),
                ShellData::I64(items) => st_new_vector_arm!(items, cosine, f64, Cosine, I64),
                ShellData::U8(items) => st_new_vector_arm!(items, cosine, f32, Cosine, U8),
                ShellData::U16(items) => st_new_vector_arm!(items, cosine, f32, Cosine, U16),
                ShellData::U32(items) => st_new_vector_arm!(items, cosine, f32, Cosine, U32),
                ShellData::U64(items) => st_new_vector_arm!(items, cosine, f64, Cosine, U64),
            },
        }
    }

    /// Search the tree with the given queries and algorithms.
    pub fn search<P: AsRef<Path> + core::fmt::Debug>(&self, queries: ShellData, algorithms: &[ShellCakes], out_path: P) -> Result<(), String> {
        match self {
            Self::Lcs(tree) => match queries {
                ShellData::String(queries) => {
                    let queries = queries.iter().map(|(_, seq)| seq.clone()).collect::<Vec<_>>();
                    search(tree, &queries, algorithms, out_path)
                }
                _ => Err("LCS tree can only be searched with string queries".to_string()),
            },
            Self::Levenshtein(tree) => match queries {
                ShellData::String(queries) => {
                    let queries = queries.iter().map(|(_, seq)| seq.clone()).collect::<Vec<_>>();
                    search(tree, &queries, algorithms, out_path)
                }
                _ => Err("Levenshtein tree can only be searched with string queries".to_string()),
            },
            Self::Euclidean(tree) => match queries {
                ShellData::String(_) => Err("Euclidean tree cannot be searched with string queries".to_string()),
                _ => tree.search(queries, algorithms, out_path),
            },
            Self::Cosine(tree) => match queries {
                ShellData::String(_) => Err("Cosine tree cannot be searched with string queries".to_string()),
                _ => tree.search(queries, algorithms, out_path),
            },
        }
    }

    /// Creates a path for the tree file based on the metric and optional suffix.
    pub fn tree_file_path<P: AsRef<Path>>(&self, out_dir: P, suffix: Option<&str>) -> PathBuf {
        let suffix = match self {
            Self::Lcs(_) => suffix.map_or_else(|| "lcs".to_string(), |s| format!("{s}-lcs")),
            Self::Levenshtein(_) => suffix.map_or_else(|| "lev".to_string(), |s| format!("{s}-lev")),
            Self::Euclidean(_) => suffix.map_or_else(|| "euc".to_string(), |s| format!("{s}-euc")),
            Self::Cosine(_) => suffix.map_or_else(|| "cos".to_string(), |s| format!("{s}-cos")),
        };
        out_dir.as_ref().join(format!("tree-{suffix}.bin"))
    }

    /// Saves the tree to the specified path using bincode.
    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        match self {
            Self::Lcs(tree) => {
                let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
                tree.encode::<DATABUF_DEFAULT>(&mut file).map_err(|e| e.to_string())
            }
            Self::Levenshtein(tree) => {
                let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
                tree.encode::<DATABUF_DEFAULT>(&mut file).map_err(|e| e.to_string())
            }
            Self::Euclidean(tree) => tree.write_to(path),
            Self::Cosine(tree) => tree.write_to(path),
        }
    }

    /// Reads a tree from the specified input directory using bincode.
    pub fn read_from<P: AsRef<Path>>(tree_path: P, metric: &Metric) -> Result<Self, String> {
        let tree_path = tree_path.as_ref();

        if !tree_path.exists() {
            return Err(format!("Tree file '{tree_path:?}' does not exist"));
        }

        if !tree_path.is_file() {
            return Err(format!("Tree path '{tree_path:?}' is not a file"));
        }

        if tree_path.extension().is_some_and(|ext| ext != "bin") {
            return Err(format!("Tree file '{tree_path:?}' does not have a .bin extension"));
        }

        match metric {
            Metric::Lcs => {
                let bytes = std::fs::read(tree_path).map_err(|e| e.to_string())?;
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(&bytes).map_err(|e| e.to_string())?;
                Ok(ShellTree::Lcs(tree.with_metric(lcs)))
            }
            Metric::Levenshtein => {
                let sz_device = stringzilla::szs::DeviceScope::default().map_err(|e| e.to_string())?;
                let sz_engine = stringzilla::szs::LevenshteinDistances::new(&sz_device, 0, 1, 1, 1).map_err(|e| e.to_string())?;
                let metric = move |a: &MusalsSequence, b: &MusalsSequence| -> u32 {
                    let distances = sz_engine.compute(&sz_device, &[a.as_ref()], &[b.as_ref()]).unwrap();
                    distances[0] as u32
                };
                let metric = Box::new(metric) as Box<dyn Fn(&MusalsSequence, &MusalsSequence) -> u32 + Send + Sync>;

                let bytes = std::fs::read(tree_path).map_err(|e| e.to_string())?;
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(&bytes).map_err(|e| e.to_string())?;
                Ok(ShellTree::Levenshtein(tree.with_metric(metric)))
            }
            Metric::Euclidean => Ok(ShellTree::Euclidean(VectorTree::read_from(tree_path)?)),
            Metric::Cosine => Ok(ShellTree::Cosine(VectorTree::read_from(tree_path)?)),
        }
    }
}

impl VectorTree {
    /// Saves the tree to the specified path using bitcode.
    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let (mut bytes, dtype) = match self {
            Self::F32(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "f4"),
            Self::F64(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "f8"),
            Self::I8(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "i1"),
            Self::I16(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "i2"),
            Self::I32(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "i4"),
            Self::I64(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "i8"),
            Self::U8(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "u1"),
            Self::U16(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "u2"),
            Self::U32(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "u4"),
            Self::U64(tree) => (tree.to_bytes::<DATABUF_DEFAULT>(), "u8"),
        };
        bytes.extend_from_slice(dtype.as_bytes());
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    /// Reads a VectorTree from the specified path using bitcode.
    pub fn read_from<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let dtype = std::str::from_utf8(&bytes[bytes.len() - 2..]).map_err(|e| e.to_string())?;

        let bytes = &bytes[..bytes.len() - 2];
        let tree = match dtype {
            "f4" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::F32(tree.with_metric(euclidean))
            }
            "f8" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::F64(tree.with_metric(euclidean))
            }
            "i1" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::I8(tree.with_metric(euclidean))
            }
            "i2" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::I16(tree.with_metric(euclidean))
            }
            "i4" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::I32(tree.with_metric(euclidean))
            }
            "i8" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::I64(tree.with_metric(euclidean))
            }
            "u1" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::U8(tree.with_metric(euclidean))
            }
            "u2" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::U16(tree.with_metric(euclidean))
            }
            "u4" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::U32(tree.with_metric(euclidean))
            }
            "u8" => {
                let tree = Tree::from_bytes::<DATABUF_DEFAULT>(bytes).map_err(|e| e.to_string())?;
                Self::U64(tree.with_metric(euclidean))
            }
            _ => return Err("Unsupported data type in tree file".to_string()),
        };

        Ok(tree)
    }

    /// Search the tree with the given queries and algorithms.
    pub fn search<P: AsRef<Path> + core::fmt::Debug>(&self, queries: ShellData, algorithms: &[ShellCakes], out_path: P) -> Result<(), String> {
        match (self, queries) {
            (Self::F32(tree), ShellData::F32(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::F64(tree), ShellData::F64(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I8(tree), ShellData::I8(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I16(tree), ShellData::I16(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I32(tree), ShellData::I32(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I64(tree), ShellData::I64(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U8(tree), ShellData::U8(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U16(tree), ShellData::U16(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U32(tree), ShellData::U32(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U64(tree), ShellData::U64(queries)) => search(tree, &queries, algorithms, out_path),
            _ => Err("Query data type does not match vectors type in tree".to_string()),
        }
    }
}

impl std::fmt::Display for VectorTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorTree::F32(_) => write!(f, "VectorTree<F32>"),
            VectorTree::F64(_) => write!(f, "VectorTree<F64>"),
            VectorTree::I8(_) => write!(f, "VectorTree<I8>"),
            VectorTree::I16(_) => write!(f, "VectorTree<I16>"),
            VectorTree::I32(_) => write!(f, "VectorTree<I32>"),
            VectorTree::I64(_) => write!(f, "VectorTree<I64>"),
            VectorTree::U8(_) => write!(f, "VectorTree<U8>"),
            VectorTree::U16(_) => write!(f, "VectorTree<U16>"),
            VectorTree::U32(_) => write!(f, "VectorTree<U32>"),
            VectorTree::U64(_) => write!(f, "VectorTree<U64>"),
        }
    }
}

impl std::fmt::Display for ShellTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellTree::Lcs(_) => write!(f, "LcsStringTree<U32>"),
            ShellTree::Levenshtein(_) => write!(f, "LevenshteinStringTree<U32>"),
            ShellTree::Euclidean(tree) => write!(f, "Euclidean{tree}"),
            ShellTree::Cosine(tree) => write!(f, "Cosine{tree}"),
        }
    }
}

fn search<Id, I, T, A, M, P>(tree: &Tree<Id, I, T, A, M>, queries: &[I], algs: &[ShellCakes], out_path: P) -> Result<(), String>
where
    T: DistanceValue + 'static,
    M: Fn(&I, &I) -> T,
    P: AsRef<Path> + core::fmt::Debug,
    <T as core::str::FromStr>::Err: std::fmt::Display,
{
    let mut all_results = SearchResults { results: Vec::new() };

    for (i, query) in queries.iter().enumerate() {
        ftlog::info!("Processing query {i}");

        let mut query_result = QueryResult {
            query_index: i,
            algorithms: Vec::new(),
        };

        for alg in algs {
            let alg = alg.get()?;
            let result = alg.search(tree, query);
            ftlog::info!("Result {}: {result:?}", alg.name());

            // Convert result to f64 for serialization consistency
            let neighbors: Vec<(usize, f64)> = result.into_iter().map(|(idx, dist)| (idx, dist.to_f64().unwrap())).collect();

            query_result.algorithms.push(AlgorithmResult {
                algorithm: alg.name(),
                neighbors,
            });
        }

        all_results.results.push(query_result);
    }

    OutputFormat::write(out_path, &all_results)
}
