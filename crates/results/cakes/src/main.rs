#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

use std::path::PathBuf;

use abd_clam::{
    adapter::ParBallAdapter,
    cakes::{CodecData, Decompressible, SquishyBall},
    partition::ParPartition,
    Ball, Cluster, Dataset, Metric, MetricSpace, Permutable,
};
use clap::Parser;
use readers::FlatGenomic;

mod readers;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset type
    #[arg(short, long)]
    dataset: readers::Datasets,

    /// Path to the dataset.
    #[arg(short, long)]
    path: PathBuf,

    /// Path to the output directory.
    #[arg(short, long)]
    out_dir: PathBuf,
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), String> {
    let args = Args::parse();

    mt_logger::mt_new!(
        Some("cakes-greengenes.log"),
        mt_logger::Level::Debug,
        mt_logger::OutputStream::Both
    );

    // Check that the output directory exists
    if !args.out_dir.exists() {
        return Err(format!("Output directory {:?} does not exist!", args.out_dir));
    }
    if !args.out_dir.is_dir() {
        return Err(format!("{:?} is not a directory!", args.out_dir));
    }
    let out_dir = args.out_dir.canonicalize().map_err(|e| e.to_string())?;
    mt_logger::mt_log!(mt_logger::Level::Info, "Output directory: {out_dir:?}");

    // Construct the path for the serialized dataset
    let original_path = args.path.canonicalize().map_err(|e| e.to_string())?;
    let flat_vec_path = args.out_dir.join("greengenes.flat_data");

    let start = std::time::Instant::now();

    // Read the dataset
    let (data, end) = if flat_vec_path.exists() {
        let data: FlatGenomic =
            bincode::deserialize_from(std::fs::File::open(&flat_vec_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Deserialized {:?} dataset from {flat_vec_path:?} with {} sequences.",
            args.dataset,
            data.cardinality()
        );
        (data, end)
    } else {
        let data = args.dataset.read(&original_path)?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Read dataset from {original_path:?} with {} sequences.",
            data.cardinality()
        );
        let serde_start = std::time::Instant::now();
        bincode::serialize_into(std::fs::File::create(&flat_vec_path).map_err(|e| e.to_string())?, &data)
            .map_err(|e| e.to_string())?;
        let serde_end = serde_start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Serialized dataset to {flat_vec_path:?} in {:.6} seconds.",
            serde_end.as_secs_f64()
        );
        (data, end)
    };

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Read the raw data in {:.6} seconds.",
        end.as_secs_f64()
    );

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Working with {:?} Dataset with {} sequences in {:?} dims.",
        args.dataset,
        data.cardinality(),
        data.dimensionality_hint()
    );

    let data = {
        let lev = |x: &String, y: &String| distances::strings::levenshtein(x, y);
        let metric = Metric::new(lev, true);
        let mut data = data;
        data.set_metric(metric);
        data
    };

    let ball_path = out_dir.join("greengenes.ball");
    let start = std::time::Instant::now();
    let ball = if ball_path.exists() {
        let ball: Ball<_, _, _> = bincode::deserialize_from(std::fs::File::open(&ball_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Deserialized BallTree from {ball_path:?} in {:.6} seconds.",
            end.as_secs_f64()
        );
        ball
    } else {
        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let seed = Some(42);
        let ball = Ball::par_new_tree(&data, &criteria, seed);
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Built BallTree in {:.6} seconds.",
            end.as_secs_f64()
        );

        let start = std::time::Instant::now();
        bincode::serialize_into(std::fs::File::create(&ball_path).map_err(|e| e.to_string())?, &ball)
            .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Serialized BallTree to {ball_path:?} in {:.6} seconds.",
            end.as_secs_f64()
        );

        ball
    };

    let subtree_cardinality = ball.subtree().len();
    mt_logger::mt_log!(mt_logger::Level::Info, "BallTree has {subtree_cardinality} clusters.");

    let metadata = data.metadata().to_vec();
    let squishy_ball_path = out_dir.join("greengenes.squishy_ball");
    let codec_data_path = out_dir.join("greengenes.codec_data");
    let start = std::time::Instant::now();
    let (squishy_ball, codec_data) = if squishy_ball_path.exists() && codec_data_path.exists() {
        let squishy_ball: SquishyBall<_, _, _, _, _> =
            bincode::deserialize_from(std::fs::File::open(&squishy_ball_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Deserialized SquishyBall from {squishy_ball_path:?} in {:.6} seconds.",
            end.as_secs_f64()
        );

        let start = std::time::Instant::now();
        let mut codec_data: CodecData<String, u32, usize> =
            bincode::deserialize_from(std::fs::File::open(&codec_data_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Deserialized CodecData from {codec_data_path:?} in {:.6} seconds.",
            end.as_secs_f64()
        );

        codec_data.set_metric(data.metric().clone());
        let codec_data = codec_data.post_deserialization(data.permutation(), metadata)?;

        (squishy_ball, codec_data)
    } else {
        let (squishy_ball, codec_data) = SquishyBall::par_from_ball_tree(ball, data);
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Built SquishyBall in {:.6} seconds.",
            end.as_secs_f64()
        );
        let codec_data = codec_data.with_metadata(metadata)?;

        let start = std::time::Instant::now();
        bincode::serialize_into(
            std::fs::File::create(&squishy_ball_path).map_err(|e| e.to_string())?,
            &squishy_ball,
        )
        .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Serialized SquishyBall to {squishy_ball_path:?} in {:.6} seconds.",
            end.as_secs_f64()
        );

        let start = std::time::Instant::now();
        bincode::serialize_into(
            std::fs::File::create(&codec_data_path).map_err(|e| e.to_string())?,
            &codec_data,
        )
        .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Serialized CodecData to {codec_data_path:?} in {:.6} seconds.",
            end.as_secs_f64()
        );

        (squishy_ball, codec_data)
    };

    let squishy_ball_subtree_cardinality = squishy_ball.subtree().len();
    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "SquishyBall has {squishy_ball_subtree_cardinality} clusters."
    );

    let num_leaf_bytes = codec_data.leaf_bytes().len();
    mt_logger::mt_log!(mt_logger::Level::Info, "CodecData has {num_leaf_bytes} leaf bytes.");

    mt_logger::mt_flush!().map_err(|e| e.to_string())
}
