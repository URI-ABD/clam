//! Tree formats supported in the CLI.

use std::path::Path;

use abd_clam::{Ball, Dataset, ParPartition, cakes::PermutedBall};

use crate::{
    data::ShellData,
    metrics::{Metric, cosine, euclidean, levenshtein},
};

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
    pub fn new(inp_data: ShellData, metric: &Metric, permuted: bool) -> Result<(ShellTree, ShellData), String> {
        // TODO Najib: Implement a macro to handle the match arms more elegantly.
        match inp_data {
            ShellData::String(data) => match metric {
                Metric::Levenshtein => {
                    let (mut data, mut metadata): (Vec<_>, Vec<_>) = data.into_iter().unzip();
                    let ball = Ball::par_new_tree(&data, &levenshtein, &|_| true);
                    if permuted {
                        let (ball, permutation) = PermutedBall::par_from_cluster_tree(ball, &mut data);
                        let ball = Self::PermutedBall(ShellPermutedBall::String(ball));

                        metadata.permute(&permutation);

                        let data = ShellData::String(data.into_iter().zip(metadata).collect());
                        Ok((ball, data))
                    } else {
                        Ok((
                            Self::Ball(ShellBall::String(ball)),
                            ShellData::String(data.into_iter().zip(metadata).collect()),
                        ))
                    }
                }
                _ => Err(format!("Metric {} cannot be used for string data", metric.name())),
            },
            ShellData::F32(mut data) => match metric {
                Metric::Levenshtein => Err("Levenshtein metric cannot be used for vector data".to_string()),
                Metric::Euclidean => {
                    let ball = Ball::par_new_tree(&data, &euclidean, &|_| true);
                    if permuted {
                        let (ball, _) = PermutedBall::par_from_cluster_tree(ball, &mut data);
                        let ball = Self::PermutedBall(ShellPermutedBall::F32(ball));
                        let data = ShellData::F32(data);
                        Ok((ball, data))
                    } else {
                        Ok((Self::Ball(ShellBall::F32(ball)), ShellData::F32(data)))
                    }
                }
                Metric::Cosine => {
                    let ball = Ball::par_new_tree(&data, &cosine, &|_| true);
                    if permuted {
                        let (ball, _) = PermutedBall::par_from_cluster_tree(ball, &mut data);
                        let ball = Self::PermutedBall(ShellPermutedBall::F32(ball));
                        let data = ShellData::F32(data);
                        Ok((ball, data))
                    } else {
                        Ok((Self::Ball(ShellBall::F32(ball)), ShellData::F32(data)))
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
