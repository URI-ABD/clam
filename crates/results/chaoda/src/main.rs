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

use clap::Parser;

use abd_clam::{
    chaoda::{GraphAlgorithm, TrainableMetaMlModel, TrainableSmc},
    dataset::AssociatesMetadata,
    metric::{Euclidean, Manhattan},
    Ball, Cluster, Dataset,
};
use distances::Number;

mod data;
mod utils;

/// Reproducible results for the CAKES and panCAKES papers.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the directory containing the datasets.
    #[arg(short('i'), long)]
    data_dir: PathBuf,

    /// Minimum depth of the clusters to use for making graphs.
    #[arg(short('d'), long, default_value = "4")]
    min_depth: usize,

    /// The number of epochs to train the model for.
    #[arg(short('e'), long, default_value = "10")]
    num_epochs: usize,

    /// Whether to use a pre-trained model.
    #[arg(short('p'), long)]
    use_pre_trained: bool,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,
}

/// The main function.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn main() -> Result<(), String> {
    #[allow(clippy::useless_format)]
    let log_name = format!("chaoda-training");
    let (_guard, log_path) = utils::configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    let args = Args::parse();

    let data_dir = std::fs::canonicalize(args.data_dir).map_err(|e| e.to_string())?;
    ftlog::info!("Reading datasets from: {data_dir:?}");

    let out_dir = args.out_dir.unwrap_or_else(|| data_dir.join("results"));
    if !out_dir.exists() {
        std::fs::create_dir(&out_dir).map_err(|e| e.to_string())?;
    }
    let out_dir = std::fs::canonicalize(out_dir).map_err(|e| e.to_string())?;
    ftlog::info!("Saving results to: {out_dir:?}");

    // Build path to pre-trained model
    let model_path = out_dir.join("chaoda.bin");
    ftlog::info!("Model path: {model_path:?}");

    let use_pre_trained = args.use_pre_trained && model_path.exists();
    if use_pre_trained {
        ftlog::info!("Using pre-trained model");
    } else {
        ftlog::info!("Training model from scratch");
    }

    // Set some parameters for tree building
    let seed = Some(42);
    let train_datasets = data::Data::read_train_data(&data_dir)?;
    let criteria = default_criteria(&train_datasets);
    let labels = {
        let mut labels = Vec::new();
        for data in &train_datasets {
            labels.push(data.metadata().to_vec());
        }
        labels
            .try_into()
            .unwrap_or_else(|_| unreachable!("We have labels for each dataset."))
    };
    let depths = (args.min_depth..).step_by(5).take(5).collect::<Vec<_>>();

    ftlog::info!("Training datasets:");
    for d in &train_datasets {
        ftlog::info!("{}", d.name());
    }

    let [model_euc, model_man] = if use_pre_trained {
        // Load the pre-trained CHAODA model
        ftlog::info!("Loading pre-trained model from: {model_path:?}");
        bincode::deserialize_from(std::fs::File::open(&model_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        // Create a Chaoda trainer
        let meta_ml_models = TrainableMetaMlModel::default_models();
        let graph_algorithms = GraphAlgorithm::default_algorithms();

        let mut smc_euc = TrainableSmc::new(&meta_ml_models, &graph_algorithms);
        let trees = smc_euc.par_create_trees(&train_datasets, &criteria, &Euclidean, seed);
        let trained_euc = smc_euc.par_train(
            &train_datasets,
            &Euclidean,
            &trees,
            &labels,
            args.min_depth,
            &depths,
            args.num_epochs,
        )?;
        ftlog::info!(
            "Completed training for {} epochs with Euclidean metric",
            args.num_epochs
        );

        let mut smc_man = TrainableSmc::new(&meta_ml_models, &graph_algorithms);
        let trees = smc_man.par_create_trees(&train_datasets, &criteria, &Manhattan, seed);
        let trained_man = smc_man.par_train(
            &train_datasets,
            &Manhattan,
            &trees,
            &labels,
            args.min_depth,
            &depths,
            args.num_epochs,
        )?;
        ftlog::info!(
            "Completed training for {} epochs with Manhattan metric",
            args.num_epochs
        );

        let trained_models = [trained_euc, trained_man];

        // Save the trained model
        ftlog::info!("Saving model to: {model_path:?}");
        bincode::serialize_into(
            std::fs::File::create(&model_path).map_err(|e| e.to_string())?,
            &trained_models,
        )
        .map_err(|e| e.to_string())?;
        ftlog::info!("Model saved to: {model_path:?}");

        trained_models
    };

    // Print the ROC scores for all datasets
    for data in data::Data::read_all(&data_dir)? {
        ftlog::info!("Starting evaluation for: {}", data.name());

        let labels = data.metadata().to_vec();
        let criteria = |c: &Ball<_>| c.cardinality() > 10;

        let roc_score = model_euc.par_evaluate(&data, &labels, &Euclidean, &criteria, seed, args.min_depth, 0.02);
        ftlog::info!(
            "Dataset: {}, Metric: Euclidean ROC-AUC score: {roc_score:.6}",
            data.name()
        );

        let roc_score = model_man.par_evaluate(&data, &labels, &Manhattan, &criteria, seed, args.min_depth, 0.02);
        ftlog::info!(
            "Dataset: {}, Metric: Manhattan ROC-AUC score: {roc_score:.6}",
            data.name()
        );
    }

    Ok(())
}

/// Returns the default partitioning criteria, repeated `N` times.
fn default_criteria<A, T: Number, const N: usize>(_: &[A; N]) -> [impl Fn(&Ball<T>) -> bool; N] {
    let mut criteria = Vec::with_capacity(N);
    for _ in 0..N {
        criteria.push(|c: &Ball<_>| c.cardinality() > 10);
    }
    criteria
        .try_into()
        .unwrap_or_else(|_| unreachable!("We have a criterion for each dataset."))
}
