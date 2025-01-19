//! Building, saving and loading trees for benchmarking.

use abd_clam::{
    cakes::PermutedBall,
    cluster::{adapter::ParBallAdapter, BalancedBall, Csv, ParClusterIO, ParPartition},
    dataset::DatasetIO,
    Ball, Dataset, FlatVec,
};
use distances::Number;

use crate::metric::ParCountingMetric;

/// The seven output paths for a given dataset.
pub struct AllPaths {
    /// The output directory.
    pub out_dir: std::path::PathBuf,
    /// The path to the `Ball` tree.
    pub ball: std::path::PathBuf,
    /// The path to the data.
    pub data: std::path::PathBuf,
    /// The path to the `BalancedBall` tree.
    pub balanced_ball: std::path::PathBuf,
    /// The path to the permuted `Ball` tree.
    pub permuted_ball: std::path::PathBuf,
    /// The path to the permuted `BalancedBall` tree.
    pub permuted_balanced_ball: std::path::PathBuf,
    /// The path to the permuted data.
    pub permuted_data: std::path::PathBuf,
    /// The path to the permuted balanced data.
    pub permuted_balanced_data: std::path::PathBuf,
}

impl AllPaths {
    /// Creates a new `AllPaths` instance.
    pub fn new<P: AsRef<std::path::Path>>(out_dir: &P, data_name: &str) -> Self {
        Self {
            out_dir: out_dir.as_ref().to_path_buf(),
            ball: out_dir.as_ref().join(format!("{data_name}.ball")),
            data: out_dir.as_ref().join(format!("{data_name}.flat_vec")),
            balanced_ball: out_dir.as_ref().join(format!("{data_name}.balanced_ball")),
            permuted_ball: out_dir.as_ref().join(format!("{data_name}.permuted_ball")),
            permuted_balanced_ball: out_dir.as_ref().join(format!("{data_name}.permuted_balanced_ball")),
            permuted_data: out_dir.as_ref().join(format!("{data_name}-permuted.flat_vec")),
            permuted_balanced_data: out_dir.as_ref().join(format!("{data_name}-permuted_balanced.flat_vec")),
        }
    }

    /// Whether all paths exist.
    pub fn all_exist(&self, balanced: bool, permuted: bool) -> bool {
        let mut base = self.ball.exists() && self.data.exists();

        if balanced {
            base = base && self.balanced_ball.exists();
        }
        if permuted {
            base = base && self.permuted_ball.exists() && self.permuted_data.exists();
            if balanced {
                base = base && self.permuted_balanced_ball.exists() && self.permuted_balanced_data.exists();
            }
        }

        base
    }
}

/// Builds all types of trees for the given dataset.
pub fn build_all<P, I, T, M, Me>(
    out_dir: &P,
    data: &FlatVec<I, Me>,
    metric: &M,
    seed: Option<u64>,
    build_permuted: bool,
    build_balanced: bool,
    depth_stride: Option<usize>,
) -> Result<(), String>
where
    P: AsRef<std::path::Path>,
    I: Send + Sync + Clone + bitcode::Encode + bitcode::Decode,
    T: Number + bitcode::Encode + bitcode::Decode,
    M: ParCountingMetric<I, T>,
    Me: Send + Sync + Clone + bitcode::Encode + bitcode::Decode,
{
    ftlog::info!("Building all trees for {}...", data.name());
    let all_paths = AllPaths::new(out_dir, data.name());

    if !all_paths.data.exists() {
        ftlog::info!("Writing data to {:?}...", all_paths.data);
        data.write_to(&all_paths.data)?;
    }

    ftlog::info!("Building Ball...");
    metric.reset_count();
    let ball = depth_stride.map_or_else(
        || Ball::par_new_tree(data, metric, &|_| true, seed),
        |depth_stride| Ball::par_new_tree_iterative(data, metric, &|_| true, seed, depth_stride),
    );
    ftlog::info!("Built Ball by calculating {} distances.", metric.count());

    ftlog::info!("Writing Ball to {:?}...", all_paths.ball);
    ball.par_write_to(&all_paths.ball)?;
    let csv_path = out_dir.as_ref().join(format!("{}-ball.csv", data.name()));
    ball.write_to_csv(&csv_path)?;

    if build_permuted {
        ftlog::info!("Building Permuted Ball...");
        let (ball, data) = PermutedBall::par_from_ball_tree(ball, data.clone(), metric);

        ftlog::info!("Writing Permuted Ball to {:?}...", all_paths.permuted_ball);
        ball.par_write_to(&all_paths.permuted_ball)?;
        let csv_path = out_dir.as_ref().join(format!("{}-permuted-ball.csv", data.name()));
        ball.write_to_csv(&csv_path)?;

        ftlog::info!("Writing Permuted data to {:?}...", all_paths.permuted_data);
        data.write_to(&all_paths.permuted_data)?;
    }

    if build_balanced {
        ftlog::info!("Building Balanced Ball...");
        metric.reset_count();
        let ball = depth_stride
            .map_or_else(
                || BalancedBall::par_new_tree(data, metric, &|_| true, seed),
                |depth_stride| BalancedBall::par_new_tree_iterative(data, metric, &|_| true, seed, depth_stride),
            )
            .into_ball();
        ftlog::info!("Built Balanced Ball by calculating {} distances.", metric.count());

        ftlog::info!("Writing Balanced Ball to {:?}...", all_paths.balanced_ball);
        ball.par_write_to(&all_paths.balanced_ball)?;
        let csv_path = out_dir.as_ref().join(format!("{}-balanced-ball.csv", data.name()));
        ball.write_to_csv(&csv_path)?;

        if build_permuted {
            ftlog::info!("Building Permuted Balanced Ball...");
            let (ball, data) = PermutedBall::par_from_ball_tree(ball, data.clone(), metric);

            ftlog::info!(
                "Writing Permuted Balanced Ball to {:?}...",
                all_paths.permuted_balanced_ball
            );
            ball.par_write_to(&all_paths.permuted_balanced_ball)?;
            let csv_path = out_dir
                .as_ref()
                .join(format!("{}-permuted-balanced-ball.csv", data.name()));
            ball.write_to_csv(&csv_path)?;

            ftlog::info!(
                "Writing Permuted Balanced data to {:?}...",
                all_paths.permuted_balanced_data
            );
            data.write_to(&all_paths.permuted_balanced_data)?;
        }
    }
    ftlog::info!("Built all trees for {}.", data.name());

    Ok(())
}
