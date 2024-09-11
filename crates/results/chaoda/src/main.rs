//! Reproduce the CHAODA results in Rust.

use std::path::Path;

use abd_clam::{Ball, Chaoda, Cluster, Dataset, Metric};

mod data;

fn main() -> Result<(), String> {
    // mt_new!(None, Level::Info, OutputStream::StdOut);

    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        return Err(format!(
            "Expected two input parameters, the directory where dataset will be read from and whether to use a pre-trained model (if found). Got {args:?} instead",
        )
        );
    }

    // Parse args[1] into path object
    let data_dir = Path::new(&args[1]);
    let data_dir = std::fs::canonicalize(data_dir).map_err(|e| e.to_string())?;
    // mt_log!(Level::Info, "Reading datasets from: {data_dir:?}");

    // Parse args[2] into boolean
    let use_pre_trained = args[2].parse::<String>().map_err(|e| e.to_string())?;
    let use_pre_trained = {
        match use_pre_trained.as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(format!(
                    "Invalid value for use_pre_trained: {use_pre_trained}. Expected 'true' or 'false'"
                ))
            }
        }
    };
    // Build path to pre-trained model
    let model_path = data_dir.join("pre-trained-chaoda-model.bin");
    let use_pre_trained = use_pre_trained && model_path.exists();

    // Set some parameters for tree building
    let seed = Some(42);

    let metrics = [
        Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::euclidean(x, y), false).with_name("euclidean"),
        Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::manhattan(x, y), false).with_name("manhattan"),
        // Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::cosine(x, y), false).with_name("cosine"),
        // Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::canberra(x, y), false).with_name("canberra"),
        // Metric::new(|x: &Vec<f64>, y: &Vec<f64>| distances::vectors::bray_curtis(x, y), false).with_name("bray_curtis"),
    ];

    let mut train_datasets = data::Data::read_paper_train(&data_dir)?;
    let criteria = {
        let mut criteria = Vec::new();
        for _ in 0..train_datasets.len() {
            criteria.push(|c: &Ball<_, _, _>| c.cardinality() > 1);
        }
        criteria
            .try_into()
            .unwrap_or_else(|_| unreachable!("We have a criterion for each dataset."))
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

    // mt_log!(Level::Info, "Training datasets:");
    // for d in &train_datasets {
    //     mt_log!(Level::Info, "{}", d.name());
    // }

    let model = if use_pre_trained {
        // Load the pre-trained CHAODA model
        // mt_log!(Level::Info, "Loading pre-trained model from: {model_path:?}");
        Chaoda::load(&model_path, &metrics)?
    } else {
        // Train the CHAODA model
        let num_epochs = 16;
        let trees = Chaoda::par_new_trees(&mut train_datasets, &criteria, &metrics, seed);
        let mut model = Chaoda::new(&metrics, None, 4);
        model.par_train(&mut train_datasets, &trees, &labels, num_epochs, None)?;
        // mt_log!(Level::Info, "Training complete");
        model.save(&model_path)?;
        // mt_log!(Level::Info, "Model saved to: {model_path:?}");
        model
    };

    // Print the ROC scores for all datasets
    for data in data::Data::read_all(&data_dir)? {
        // mt_log!(Level::Info, "Starting evaluation for: {}", data.name());
        let mut data = [data];
        let criteria = [|c: &Ball<_, _, _>| c.cardinality() > 1];
        let trees = Chaoda::par_new_trees(&mut data, &criteria, &metrics, seed);
        let labels = data[0].metadata().to_vec();

        let mut data = data.into_iter().next().unwrap();
        let trees = trees.into_iter().next().unwrap();
        let roc_score = model.par_evaluate(&mut data, &trees, &labels);
        println!("Dataset: {} ROC-AUC score: {:.6}", data.name(), roc_score);
        // mt_log!(Level::Info, "Dataset: {} ROC-AUC score: {roc_score:.6}", data.name());
    }

    // mt_flush!().map_err(|e| e.to_string())

    Ok(())
}
