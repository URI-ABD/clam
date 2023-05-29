// //! CLAM Command Line Interface.
// //!
// //! # Basics
// //!
// //! CLAM operates similarly to [`git`]. There is the main entry-point `clam`, followed by a subcommand.
// //! In other words, the structure of a call to CLAM should be: `clam <args> <command> <command-specific args>`.
// //!
// //! Example call to `clam chaoda`:
// //! `cargo build --release`
// //! `./target/release/clam chaoda --mode=bench --max-tree-depth=5 --metrics euclidean --dataset=path/to/data`
// //!
// //! # CHAODA: Clustered Hierarchical Anomaly and Outlier Detection Algorithms
// //!
// //! The research paper for CHAODA can be found [`here`]: https://github.com/URI-ABD/chaoda-paper.
// //!
// //! CHAODA currently supports the following modes:
// //!
// //! * `bench`: benchmark results (roc-score and time) against a dataset.
// //! * `score`: compute outlier-scores (and time) against a dataset.
// //!
// //! [`git`]: https://git-scm.com/
//
// use clam::prelude::*;
// use eval_metrics::classification::RocCurve;
// use log::*;
// use serde_json::json;
// use serde_json::to_string;
// use simplelog::*;
// use std::io::prelude::*;
// use std::path::PathBuf;
// use std::sync::Arc;
// use structopt::clap::arg_enum;
// use structopt::StructOpt;
//
// // This is the central entry-point for defining command-line arguments.
// #[derive(Debug, StructOpt)]
// #[structopt(
//     name = "clam",
//     about = "Clustered Learning of Approximate Manifolds",
//     rename_all = "kebab-case"
// )]
// pub struct Opt {
//     /// Activate debug mode.
//     #[structopt(long, possible_values = &LogLevel::variants(), case_insensitive = true, env = "LOG_LEVEL")]
//     log_level: Option<LogLevel>,
//
//     /// Output file, stdout if not present.
//     #[structopt(long)]
//     out: Option<PathBuf>,
//
//     /// Along with writing the log to stderr, write it to a file.
//     #[structopt(long)]
//     log_file: Option<PathBuf>,
//
//     /// Subcommand to run.
//     #[structopt(subcommand)]
//     cmd: Command,
// }
//
// arg_enum! {
//     #[derive(Debug)]
//     enum LogLevel {
//         Debug,
//         Info,
//         Warning,
//         Error
//     }
// }
//
// // arg_enum handles to_string, from_string, etc.
// arg_enum! {
//     /// CHAODA Modes.
//     /// Currently only supports Bench for benchmarking.
//     #[derive(Debug)]
//     enum ChaodaMode {
//         Bench,
//         Score,
//     }
// }
//
// arg_enum! {
//     /// Supported Metrics.
//     /// Currently supports `euclidean`.
//     #[derive(Debug)]
//     enum Metric {
//         Euclidean,
//         Manhattan,
//     }
// }
//
// /// CLAM Subcommands.
// #[derive(Debug, StructOpt)]
// enum Command {
//     /// Clustered Hierarchical Anomaly and Outlier Detection Algorithms
//     Chaoda {
//         /// Runtime mode.
//         #[structopt(short, long, possible_values = &ChaodaMode::variants(), case_insensitive = true)]
//         mode: ChaodaMode,
//
//         /// Dataset to run against. (Assumes .npy extension)
//         #[structopt(short, long)]
//         dataset: PathBuf,
//
//         /// List of metrics to use.
//         #[structopt(long, possible_values = &Metric::variants(), case_insensitive = true)]
//         metrics: Vec<Metric>,
//
//         /// Optional: Maximum tree depth.
//         #[structopt(long)]
//         max_tree_depth: Option<usize>,
//
//         /// Optional: Minimum leaf size.
//         #[structopt(long)]
//         min_leaf_size: Option<usize>,
//
//         /// Optional: Minimum depth for a cluster to be selected for an optimal graph.
//         #[structopt(long)]
//         min_selection_depth: Option<usize>,
//
//         /// Optional: Whether to use a speed threshold.
//         #[structopt(long)]
//         use_speed_threshold: bool,
//     },
// }
//
// fn main() {
//     // Collect our command line arguments.
//     let opt = Opt::from_args();
//
//     // Establish logging.
//     let log_level = match opt.log_level.unwrap_or(LogLevel::Info) {
//         LogLevel::Debug => LevelFilter::Debug,
//         LogLevel::Info => LevelFilter::Info,
//         LogLevel::Warning => LevelFilter::Warn,
//         LogLevel::Error => LevelFilter::Error,
//     };
//     let term_logger = TermLogger::new(log_level, Config::default(), TerminalMode::Stderr, ColorChoice::Auto);
//     match opt.log_file {
//         Some(file) => CombinedLogger::init(vec![
//             term_logger,
//             WriteLogger::new(
//                 LevelFilter::Info,
//                 Config::default(),
//                 std::fs::File::create(file).unwrap(),
//             ),
//         ])
//         .unwrap(),
//         None => CombinedLogger::init(vec![term_logger]).unwrap(),
//     }
//
//     info!("Starting CHAODA.");
//
//     // Do whatever work we have to do, collect the result.
//     let result = match opt.cmd {
//         Command::Chaoda {
//             mode,
//             dataset,
//             metrics,
//             max_tree_depth,
//             min_leaf_size,
//             min_selection_depth,
//             use_speed_threshold,
//         } => chaoda(
//             mode,
//             dataset,
//             &metrics,
//             max_tree_depth,
//             min_leaf_size,
//             min_selection_depth,
//             use_speed_threshold,
//         ),
//     };
//
//     // Handle output (stdout or to file)
//     match result {
//         Ok(result) => match opt.out {
//             Some(path) => write!(std::fs::File::create(path).unwrap(), "{}", result).unwrap(),
//             None => println!("{:?}", result),
//         },
//         Err(error) => error!("Uh oh, something went wrong...\n{}", error),
//     }
// }
//
// /// CHAODA helper function.
// ///
// /// Encapsulates all work done by the `chaoda` subcommand.
// fn chaoda(
//     mode: ChaodaMode,
//     dataset_path: PathBuf,
//     metrics: &[Metric],
//     max_tree_depth: Option<usize>,
//     min_leaf_size: Option<usize>,
//     min_selection_depth: Option<usize>,
//     use_speed_threshold: bool,
// ) -> Result<String, String> {
//     let read_labels = matches!(mode, ChaodaMode::Bench);
//     let (data, labels) = clam::utils::readers::read_chaoda_data(dataset_path, read_labels)?;
//     let data = Arc::new(data);
//
//     let datasets: Vec<_> = metrics
//         .iter()
//         .map(|metric| {
//             let metric = metric_from_name(&metric.to_string().to_lowercase()).unwrap();
//             let dataset: Arc<dyn Dataset<f64, f64>> =
//                 Arc::new(clam::dataset::Tabular::<f64, f64>::new(Arc::clone(&data), metric, true));
//             dataset
//         })
//         .collect();
//
//     let cluster_scorers = clam::get_meta_ml_methods();
//
//     // Start a timer.
//     let now = std::time::Instant::now();
//
//     // Call chaoda::new.
//     let chaoda = clam::Chaoda::new(
//         datasets,
//         max_tree_depth,
//         min_leaf_size,
//         cluster_scorers,
//         min_selection_depth,
//         use_speed_threshold,
//     );
//
//     let time = now.elapsed().as_secs_f64();
//
//     // Return the results as a string (JSON for Bench and Score)
//     let result = match mode {
//         ChaodaMode::Bench => {
//             let roc_score = RocCurve::compute(&chaoda.scores, &labels)
//                 .map_err(|error| format!("Error: Failed to compute RocCurve. {}", error))?
//                 .auc();
//             let result = json!({"roc_score": roc_score, "time (s)": time});
//             to_string(&result).unwrap()
//         }
//         ChaodaMode::Score => {
//             let result = json!({"scores": chaoda.scores, "time (s)": time});
//             to_string(&result).unwrap()
//         }
//     };
//
//     Ok(result)
// }

fn main() {
    println!("Hello from CLAM!")
}
