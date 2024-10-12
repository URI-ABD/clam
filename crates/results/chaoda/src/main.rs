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
    chaoda::{ChaodaTrainer, GraphAlgorithm, TrainableMetaMlModel},
    Ball, Cluster, Dataset, Metric,
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

    let metrics = [
        Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::euclidean(x, y), false).with_name("euclidean"),
        Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::manhattan(x, y), false).with_name("manhattan"),
        // Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::cosine(x, y), false).with_name("cosine"),
        // Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::canberra(x, y), false).with_name("canberra"),
        // Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::bray_curtis(x, y), false).with_name("bray_curtis"),
    ];
    ftlog::info!("Using {} metrics...", metrics.len());

    let mut train_datasets = data::Data::read_paper_train(&data_dir)?;

    let criteria = {
        let mut criteria = Vec::new();
        for _ in 0..train_datasets.len() {
            criteria.push(default_criteria::<_, _, _, 2>());
        }
        criteria
            .try_into()
            .unwrap_or_else(|_| unreachable!("We have a criterion for each pair of metric and dataset."))
    };
    let labels = {
        let mut labels = Vec::new();
        for data in &train_datasets {
            labels.push(data.metadata().to_vec());
        }
        labels
            .try_into()
            .unwrap_or_else(|_| unreachable!("We have labels for each dataset."))
    };

    ftlog::info!("Training datasets:");
    for d in &train_datasets {
        ftlog::info!("{}", d.name());
    }

    let model = if use_pre_trained {
        // Load the pre-trained CHAODA model
        ftlog::info!("Loading pre-trained model from: {model_path:?}");
        bincode::deserialize_from(std::fs::File::open(&model_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        // Create a Chaoda trainer
        let meta_ml_models = TrainableMetaMlModel::default_models();
        let graph_algorithms = GraphAlgorithm::default_algorithms();
        let mut model = ChaodaTrainer::new_all_pairs(metrics.clone(), meta_ml_models, graph_algorithms);

        // Create the trees for use in training the model.
        let trees = model.par_create_trees(&mut train_datasets, &criteria, seed);

        // Train the model
        let trained_model = model.par_train(&mut train_datasets, &trees, &labels, args.min_depth, args.num_epochs)?;
        ftlog::info!("Completed training for {} epochs", args.num_epochs);

        // Save the trained model
        ftlog::info!("Saving model to: {model_path:?}");
        bincode::serialize_into(
            std::fs::File::create(&model_path).map_err(|e| e.to_string())?,
            &trained_model,
        )
        .map_err(|e| e.to_string())?;
        ftlog::info!("Model saved to: {model_path:?}");

        trained_model
    };

    let model = if use_pre_trained {
        let mut model = model;
        model.set_metrics(metrics);
        model
    } else {
        model
    };

    // Print the ROC scores for all datasets
    for mut data in data::Data::read_all(&data_dir)? {
        ftlog::info!("Starting evaluation for: {}", data.name());

        let labels = data.metadata().to_vec();
        let criteria = default_criteria::<_, _, _, 2>();
        let roc_score = model.par_evaluate(&mut data, &criteria, &labels, seed, args.min_depth);
        ftlog::info!("Dataset: {} ROC-AUC score: {roc_score:.6}", data.name());
    }

    Ok(())
}

/// Returns the default partitioning criteria, repeated `N` times.
fn default_criteria<I, U: Number, D: Dataset<I, U>, const N: usize>() -> [impl Fn(&Ball<I, U, D>) -> bool; N] {
    let mut criteria = Vec::with_capacity(N);
    for _ in 0..N {
        criteria.push(|c: &Ball<_, _, _>| c.cardinality() > 10);
    }
    criteria
        .try_into()
        .unwrap_or_else(|_| unreachable!("We have a criterion for each dataset."))
}
