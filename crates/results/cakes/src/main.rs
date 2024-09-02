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
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

use std::path::PathBuf;

use abd_clam::{
    adapter::ParBallAdapter,
    cakes::{Algorithm, CodecData, Decompressible, SquishyBall},
    partition::ParPartition,
    Ball, Cluster, Dataset, FlatVec, MetricSpace,
};
use clap::Parser;

mod metrics;
mod readers;
mod sequence;
mod tables;

use metrics::StringDistance;
use sequence::AlignedSequence;

/// The Vector of held-out queries
pub type Queries = Vec<(String, AlignedSequence)>;

/// The type of the compressible dataset.
pub type Co = FlatVec<AlignedSequence, u32, String>;
/// The type of the ball tree over the compressible dataset.
type B = Ball<AlignedSequence, u32, Co>;
/// The type of the compressed, decompressible dataset.
type Dec = CodecData<AlignedSequence, u32, String>;
/// The type of the squishy ball tree.
type SB = SquishyBall<AlignedSequence, u32, Co, Dec, B>;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset type
    #[arg(short, long)]
    dataset: readers::Datasets,

    /// The number of queries to hold out.
    #[arg(short, long)]
    num_queries: usize,

    /// Path to the input directory.
    #[arg(short, long)]
    inp_dir: PathBuf,

    /// Path to the output directory.
    #[arg(short, long)]
    out_dir: PathBuf,
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), String> {
    let args = Args::parse();

    mt_logger::mt_new!(
        Some("cakes-results.log"),
        mt_logger::Level::Debug,
        mt_logger::OutputStream::Both
    );

    // Check that the directories exist
    for dir in [&args.inp_dir, &args.out_dir] {
        if !dir.exists() {
            return Err(format!("{dir:?} does not exist!"));
        }
        if !dir.is_dir() {
            return Err(format!("{dir:?} is not a directory!"));
        }
    }

    let inp_dir = args.inp_dir.canonicalize().map_err(|e| e.to_string())?;
    mt_logger::mt_log!(mt_logger::Level::Info, "Input directory: {inp_dir:?}");

    let out_dir = args.out_dir.canonicalize().map_err(|e| e.to_string())?;
    mt_logger::mt_log!(mt_logger::Level::Info, "Output directory: {out_dir:?}");

    let ball_path = out_dir.join(args.dataset.ball_file());
    let flat_vec_path = out_dir.join(args.dataset.flat_file());
    let queries_path = out_dir.join(args.dataset.queries_file());

    let squishy_ball_path = out_dir.join(args.dataset.squishy_ball_file());
    let codec_data_path = out_dir.join(args.dataset.compressed_file());

    let extension = "csv";
    let ball_table_path = out_dir.join(args.dataset.ball_table("ball", extension));
    let pre_trim_table_path = out_dir.join(args.dataset.ball_table("pre_trim", extension));
    let squishy_ball_table_path = out_dir.join(args.dataset.ball_table("squishy_ball", extension));

    // Read the dataset
    let (data, queries) = if flat_vec_path.exists() {
        let mut data: Co = bincode::deserialize_from(std::fs::File::open(&flat_vec_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        data.set_metric(StringDistance::Hamming.metric());

        let queries: Queries = bincode::deserialize_from(std::fs::File::open(&queries_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

        (data, queries)
    } else {
        let (data, queries): (Co, Queries) = args.dataset.read_fasta(&inp_dir, args.num_queries)?;

        bincode::serialize_into(std::fs::File::create(&flat_vec_path).map_err(|e| e.to_string())?, &data)
            .map_err(|e| e.to_string())?;

        bincode::serialize_into(
            std::fs::File::create(&queries_path).map_err(|e| e.to_string())?,
            &queries,
        )
        .map_err(|e| e.to_string())?;

        (data, queries)
    };

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Working with {:?} Dataset with {} sequences in {:?} dims.",
        args.dataset,
        data.cardinality(),
        data.dimensionality_hint()
    );

    mt_logger::mt_log!(mt_logger::Level::Info, "Holding out {} queries", queries.len());

    let ball: B = if ball_path.exists() {
        bincode::deserialize_from(std::fs::File::open(&ball_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        let mut depth = 0;
        let depth_delta = 256;
        let seed = Some(42);

        let criteria = |c: &B| c.depth() < 1;
        let mut ball = Ball::par_new_tree(&data, &criteria, seed);

        while ball.leaves().into_iter().any(|c| !c.is_singleton()) {
            depth += depth_delta;
            let criteria = |c: &B| c.depth() < depth;
            ball.par_partition_further(&data, &criteria, seed);
        }

        bincode::serialize_into(std::fs::File::create(&ball_path).map_err(|e| e.to_string())?, &ball)
            .map_err(|e| e.to_string())?;

        ball
    };

    if extension == "csv" {
        tables::write_ball_csv(&ball, &ball_table_path)?;
    } else {
        tables::write_ball_table(&ball, &ball_table_path)?;
    }

    let subtree_cardinality = ball.subtree().len();
    mt_logger::mt_log!(mt_logger::Level::Info, "BallTree has {subtree_cardinality} clusters.");

    let metadata = data.metadata().to_vec();
    let (squishy_ball, codec_data): (SB, Dec) = if squishy_ball_path.exists() && codec_data_path.exists() {
        let squishy_ball: SB =
            bincode::deserialize_from(std::fs::File::open(&squishy_ball_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

        let mut codec_data: Dec =
            bincode::deserialize_from(std::fs::File::open(&codec_data_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

        codec_data.set_metric(data.metric().clone());

        (squishy_ball, codec_data)
    } else {
        let (squishy_ball, codec_data) = SquishyBall::par_from_ball_tree(ball.clone(), data.clone(), true);
        let squishy_ball: SB = squishy_ball.with_metadata_type::<String>();
        let codec_data: Dec = codec_data.with_metadata(metadata)?;

        bincode::serialize_into(
            std::fs::File::create(&squishy_ball_path).map_err(|e| e.to_string())?,
            &squishy_ball,
        )
        .map_err(|e| e.to_string())?;

        bincode::serialize_into(
            std::fs::File::create(&codec_data_path).map_err(|e| e.to_string())?,
            &codec_data,
        )
        .map_err(|e| e.to_string())?;

        (squishy_ball, codec_data)
    };

    {
        let (pre_trim_ball, _) = SquishyBall::par_from_ball_tree(ball.clone(), data.clone(), false);
        let pre_trim_ball: SB = pre_trim_ball.with_metadata_type::<String>();
        if extension == "csv" {
            tables::write_squishy_ball_csv(&pre_trim_ball, &pre_trim_table_path)?;
        } else {
            tables::write_squishy_ball_table(&pre_trim_ball, &pre_trim_table_path)?;
        }
    }

    if extension == "csv" {
        tables::write_squishy_ball_csv(&squishy_ball, &squishy_ball_table_path)?;
    } else {
        tables::write_squishy_ball_table(&squishy_ball, &squishy_ball_table_path)?;
    }

    let squishy_ball_subtree_cardinality = squishy_ball.subtree().len();
    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "SquishyBall has {squishy_ball_subtree_cardinality} clusters."
    );

    let num_leaf_bytes = codec_data.leaf_bytes().len();
    mt_logger::mt_log!(mt_logger::Level::Info, "CodecData has {num_leaf_bytes} leaf bytes.");

    mt_logger::mt_flush!().map_err(|e| e.to_string())?;

    // Note: Starting search benchmarks here

    let (_, queries): (Vec<_>, Vec<_>) = queries.into_iter().unzip();
    // let (data, codec_data) = {
    //     let metric = StringDistance::Levenshtein.metric();
    //     let mut data = data;
    //     data.set_metric(metric.clone());
    //     let mut codec_data = codec_data;
    //     codec_data.set_metric(metric);
    //     (data, codec_data)
    // };

    let algorithms = {
        let mut algorithms = Vec::new();

        for radius in [1, 5, 10] {
            algorithms.push(Algorithm::RnnLinear(radius));
            algorithms.push(Algorithm::RnnClustered(radius));
        }

        for k in [1, 10, 100] {
            algorithms.push(Algorithm::KnnLinear(k));
            algorithms.push(Algorithm::KnnRepeatedRnn(k, 2));
            algorithms.push(Algorithm::KnnBreadthFirst(k));
            algorithms.push(Algorithm::KnnDepthFirst(k));
        }

        algorithms.clear();

        algorithms
    };

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Starting search benchmarks on {} algorithms ...",
        algorithms.len()
    );

    // TODO: Remove this limit
    let queries = &queries[..10];

    for (i, alg) in algorithms.iter().enumerate() {
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "\n\nStarting {} Search ({}/{}) on Ball and FlatVec ...",
            alg.name(),
            i + 1,
            algorithms.len()
        );
        let start = std::time::Instant::now();
        let hits = alg.par_batch_par_search(&data, &ball, queries);
        let end = start.elapsed().as_secs_f32();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Finished {} Search ({}/{}) on Ball and FlatVec in {end:.6} seconds.",
            alg.name(),
            i + 1,
            algorithms.len()
        );
        let mean_num_hits = abd_clam::utils::mean::<_, f32>(&hits.into_iter().map(|h| h.len()).collect::<Vec<_>>());
        mt_logger::mt_log!(mt_logger::Level::Info, "Average number of hits was {mean_num_hits:.6}.");

        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Starting {} Search ({}/{}) on SquishyBall and CodecData ...",
            alg.name(),
            i + 1,
            algorithms.len()
        );
        let start = std::time::Instant::now();
        let hits = alg.par_batch_par_search(&codec_data, &squishy_ball, queries);
        let end = start.elapsed().as_secs_f32();
        mt_logger::mt_log!(
            mt_logger::Level::Info,
            "Finished {} Search ({}/{}) on SquishyBall and CodecData in {end:.6} seconds.",
            alg.name(),
            i + 1,
            algorithms.len()
        );
        let mean_num_hits = abd_clam::utils::mean::<_, f32>(&hits.into_iter().map(|h| h.len()).collect::<Vec<_>>());
        mt_logger::mt_log!(mt_logger::Level::Info, "Average number of hits was {mean_num_hits:.6}.");
    }

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Finished search benchmarks on {} algorithms ...",
        algorithms.len()
    );

    mt_logger::mt_flush!().map_err(|e| e.to_string())
}
