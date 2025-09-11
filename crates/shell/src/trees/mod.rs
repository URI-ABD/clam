//! Tree formats supported in the CLI.

use std::path::Path;

use abd_clam::{
    Ball,
    adapters::ParBallAdapter,
    cakes::PermutedBall,
    cluster::{BalancedBall, ParPartition},
};

use crate::{data::ShellFlatVec, metrics::ShellMetric};

#[derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)]
pub enum ShellTree {
    Ball(ShellBall),
    PermutedBall(ShellPermutedBall),
}

impl ShellTree {
    /// Creates a new tree given a dataset and a metric.
    ///
    /// # Arguments
    ///
    /// - `inp_data`: The input data to build the tree from.
    /// - `metric`: The distance metric to use for the tree.
    /// - `seed`: The random seed to use.
    /// - `balanced`: Whether to build a balanced tree.
    /// - `permuted`: Whether to apply depth-first-reordering to the data.
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
    pub fn new(
        inp_data: ShellFlatVec,
        metric: &ShellMetric,
        seed: Option<u64>,
        balanced: bool,
        permuted: bool,
    ) -> Result<(ShellTree, ShellFlatVec), String> {
        // TODO Najib: Implement a macro to handle the match arms more elegantly.
        match inp_data {
            ShellFlatVec::String(data) => match metric {
                ShellMetric::Levenshtein(metric) => {
                    let ball = if balanced {
                        BalancedBall::par_new_tree(&data, metric, &|_| true, seed).into_ball()
                    } else {
                        Ball::par_new_tree(&data, metric, &|_| true, seed)
                    };
                    if permuted {
                        let (ball, data) = PermutedBall::par_from_ball_tree(ball, data, metric);
                        let ball = Self::PermutedBall(ShellPermutedBall::String(ball));
                        let data = ShellFlatVec::String(data);
                        Ok((ball, data))
                    } else {
                        Ok((Self::Ball(ShellBall::String(ball)), ShellFlatVec::String(data)))
                    }
                }
                ShellMetric::Euclidean(_) => Err("Euclidean metric cannot be used for string data".to_string()),
                ShellMetric::Cosine(_) => Err("Cosine metric cannot be used for string data".to_string()),
            },
            ShellFlatVec::F32(data) => match metric {
                ShellMetric::Levenshtein(_) => Err("Levenshtein metric cannot be used for vector data".to_string()),
                ShellMetric::Euclidean(metric) => {
                    let ball = if balanced {
                        BalancedBall::par_new_tree(&data, metric, &|_| true, seed).into_ball()
                    } else {
                        Ball::par_new_tree(&data, metric, &|_| true, seed)
                    };
                    if permuted {
                        let (ball, data) = PermutedBall::par_from_ball_tree(ball, data, metric);
                        let ball = Self::PermutedBall(ShellPermutedBall::F32(ball));
                        let data = ShellFlatVec::F32(data);
                        Ok((ball, data))
                    } else {
                        Ok((Self::Ball(ShellBall::F32(ball)), ShellFlatVec::F32(data)))
                    }
                }
                ShellMetric::Cosine(metric) => {
                    let ball = if balanced {
                        BalancedBall::par_new_tree(&data, metric, &|_| true, seed).into_ball()
                    } else {
                        Ball::par_new_tree(&data, metric, &|_| true, seed)
                    };
                    if permuted {
                        let (ball, data) = PermutedBall::par_from_ball_tree(ball, data, metric);
                        let ball = Self::PermutedBall(ShellPermutedBall::F32(ball));
                        let data = ShellFlatVec::F32(data);
                        Ok((ball, data))
                    } else {
                        Ok((Self::Ball(ShellBall::F32(ball)), ShellFlatVec::F32(data)))
                    }
                }
            },
            _ => {
                todo!("Najib: Implement remaining match arms via macro");
            }
        }
    }

    /// Saves the tree to the specified path using bincode.
    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let contents = bitcode::encode(self).map_err(|e| e.to_string())?;
        std::fs::write(path, contents).map_err(|e| e.to_string())
    }

    /// Reads a tree from the specified path using bincode.
    pub fn read_from<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let contents = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&contents).map_err(|e| e.to_string())
    }
}

impl std::fmt::Display for ShellTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellTree::Ball(ShellBall::String(_)) => write!(f, "Ball<String>"),
            ShellTree::Ball(ShellBall::F32(_)) => write!(f, "Ball<F32>"),
            ShellTree::Ball(ShellBall::F64(_)) => write!(f, "Ball<F64>"),
            ShellTree::Ball(ShellBall::I8(_)) => write!(f, "Ball<I8>"),
            ShellTree::Ball(ShellBall::I16(_)) => write!(f, "Ball<I16>"),
            ShellTree::Ball(ShellBall::I32(_)) => write!(f, "Ball<I32>"),
            ShellTree::Ball(ShellBall::I64(_)) => write!(f, "Ball<I64>"),
            ShellTree::Ball(ShellBall::U8(_)) => write!(f, "Ball<U8>"),
            ShellTree::Ball(ShellBall::U16(_)) => write!(f, "Ball<U16>"),
            ShellTree::Ball(ShellBall::U32(_)) => write!(f, "Ball<U32>"),
            ShellTree::Ball(ShellBall::U64(_)) => write!(f, "Ball<U64>"),
            ShellTree::PermutedBall(ShellPermutedBall::String(_)) => write!(f, "PermutedBall<String>"),
            ShellTree::PermutedBall(ShellPermutedBall::F32(_)) => write!(f, "PermutedBall<F32>"),
            ShellTree::PermutedBall(ShellPermutedBall::F64(_)) => write!(f, "PermutedBall<F64>"),
            ShellTree::PermutedBall(ShellPermutedBall::I8(_)) => write!(f, "PermutedBall<I8>"),
            ShellTree::PermutedBall(ShellPermutedBall::I16(_)) => write!(f, "PermutedBall<I16>"),
            ShellTree::PermutedBall(ShellPermutedBall::I32(_)) => write!(f, "PermutedBall<I32>"),
            ShellTree::PermutedBall(ShellPermutedBall::I64(_)) => write!(f, "PermutedBall<I64>"),
            ShellTree::PermutedBall(ShellPermutedBall::U8(_)) => write!(f, "PermutedBall<U8>"),
            ShellTree::PermutedBall(ShellPermutedBall::U16(_)) => write!(f, "PermutedBall<U16>"),
            ShellTree::PermutedBall(ShellPermutedBall::U32(_)) => write!(f, "PermutedBall<U32>"),
            ShellTree::PermutedBall(ShellPermutedBall::U64(_)) => write!(f, "PermutedBall<U64>"),
        }
    }
}

#[derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)]
pub enum ShellBall {
    String(Ball<u32>),
    F32(Ball<f32>),
    F64(Ball<f64>),
    I8(Ball<i8>),
    I16(Ball<i16>),
    I32(Ball<i32>),
    I64(Ball<i64>),
    U8(Ball<u8>),
    U16(Ball<u16>),
    U32(Ball<u32>),
    U64(Ball<u64>),
}

#[derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)]
pub enum ShellPermutedBall {
    String(PermutedBall<u32, Ball<u32>>),
    F32(PermutedBall<f32, Ball<f32>>),
    F64(PermutedBall<f64, Ball<f64>>),
    I8(PermutedBall<i8, Ball<i8>>),
    I16(PermutedBall<i16, Ball<i16>>),
    I32(PermutedBall<i32, Ball<i32>>),
    I64(PermutedBall<i64, Ball<i64>>),
    U8(PermutedBall<u8, Ball<u8>>),
    U16(PermutedBall<u16, Ball<u16>>),
    U32(PermutedBall<u32, Ball<u32>>),
    U64(PermutedBall<u64, Ball<u64>>),
}
