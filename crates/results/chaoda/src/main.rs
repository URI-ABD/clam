//! Reproduce the CHAODA results in Rust.

use std::path::Path;

use abd_clam::{
    chaoda::{Chaoda, Vertex},
    Cluster, Dataset, PartitionCriteria, VecDataset,
};

mod data;

fn main() -> Result<(), String> {
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
    println!("Reading datasets from: {data_dir:?}");

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
    let criteria = PartitionCriteria::default();

    #[allow(clippy::type_complexity)]
    let named_metrics: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] = &[
        ("euclidean", |x, y| distances::vectors::euclidean(x, y)),
        ("manhattan", |x, y| distances::vectors::manhattan(x, y)),
        // ("cosine", |x, y| distances::vectors::cosine(x, y)),
        // ("canberra", |x, y| distances::vectors::canberra(x, y)),
        // ("bray_curtis", |x, y| distances::vectors::bray_curtis(x, y)),
    ];

    // Read the datasets and assign the metrics
    let train_names: &[&str] = &[
        "arrhythmia",
        "mnist",
        "pendigits",
        "satellite",
        // "shuttle",
        "thyroid",
    ];
    let train_datasets = {
        let datasets = train_names
            .iter()
            .map(|&name| {
                let (train, labels) = data::Data::new(name)?.read(&data_dir)?;
                Ok((name, (train, labels)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let datasets = datasets
            .into_iter()
            .map(|(name, (train, labels))| {
                let (metric_name, metric) = named_metrics[0];
                let train = VecDataset::new(format!("{name}-{metric_name}"), train, metric, false);
                let train = train.assign_metadata(labels)?;
                let mut datasets = vec![train];
                for &(metric_name, metric) in named_metrics.iter().skip(1) {
                    let train = datasets[0].clone_with_new_metric(metric, false, format!("{name}-{metric_name}"));
                    datasets.push(train);
                }
                Ok(datasets)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let datasets = datasets.into_iter().flatten().collect::<Vec<_>>();
        datasets
            .into_iter()
            .map(|d| {
                let labels = d.metadata().to_vec();
                (d, labels)
            })
            .collect::<Vec<_>>()
    };
    println!("Training datasets:");
    for (d, _) in train_datasets.iter() {
        println!("{}", d.name());
    }

    let model = if use_pre_trained {
        // Load the pre-trained CHAODA model
        println!("Loading pre-trained model from: {model_path:?}");
        Chaoda::load(&model_path)?
    } else {
        // Train the CHAODA model
        let num_epochs = 16;
        let mut model = Chaoda::default();
        model.train::<_, _, _, Vertex<_>, _>(&train_datasets, num_epochs, &criteria, None, seed);
        println!("Training complete");
        model.save(&model_path)?;
        println!("Model saved to: {model_path:?}");
        model
    };

    // Print the ROC scores for all datasets
    for (name, (data, labels)) in data::Data::read_all(&data_dir)? {
        println!("Starting evaluation for: {name}");

        let (metric_name, metric) = named_metrics[0];
        let dataset = VecDataset::new(format!("{name}-{metric_name}"), data, metric, false);
        let dataset = dataset.assign_metadata(labels.clone())?;
        let mut datasets = vec![dataset];

        for &(metric_name, metric) in named_metrics.iter().skip(1) {
            let dataset = datasets[0].clone_with_new_metric(metric, false, format!("{name}-{metric_name}"));
            datasets.push(dataset);
        }

        let roots = datasets
            .iter_mut()
            .map(|dataset| Vertex::new_root(dataset, seed).partition(dataset, &criteria, seed))
            .collect::<Vec<_>>();

        let scores = datasets
            .iter()
            .zip(roots.iter())
            .map(|(d, r)| model.predict(d, r))
            .collect::<Vec<_>>();
        let y_pred = Chaoda::aggregate_predictions(&scores);

        let y_true = labels
            .into_iter()
            .map(|l| if l { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let roc_auc = Chaoda::roc_auc_score(&y_true, &y_pred);
        println!("{name}: Aggregate {roc_auc:.6}");
    }

    Ok(())
}
