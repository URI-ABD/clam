use clam::prelude::*;

mod readers;

#[allow(clippy::single_element_loop)]
fn search(data_name: &str, metric_name: &str, num_runs: usize) -> Result<(), String> {
    log::info!("");

    let (features, _) = readers::read_anomaly_data(data_name, true)?;
    let dataset = clam::Tabular::new(&features, data_name.to_string());

    let queries = (0..1000)
        .map(|i| dataset.get(i % dataset.cardinality()))
        .collect::<Vec<_>>();
    let num_queries = queries.len();
    log::info!(
        "Running {}-{} data with shape ({}, {}) and {} queries ...",
        data_name,
        metric_name,
        dataset.cardinality(),
        dataset.dimensionality(),
        num_queries,
    );

    let metric = metric_from_name::<f32, f32>(metric_name, false).unwrap();
    let space = clam::TabularSpace::new(&dataset, metric.as_ref(), false);
    let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1);

    let start = std::time::Instant::now();
    let cakes = clam::CAKES::new(&space).build(&partition_criteria);
    let build_time = start.elapsed().as_secs_f64();
    log::info!(
        "Built tree to a depth of {} in {:.2e} seconds ...",
        cakes.depth(),
        build_time
    );

    // for k in [1, 10, 100] {
    for k in [100] {
        if k > dataset.cardinality() {
            continue;
        }
        log::info!("Using k = {} ...", k);

        let start = std::time::Instant::now();
        (0..num_runs).for_each(|_| {
            cakes.batch_knn_search(&queries, k);
        });
        let time = start.elapsed().as_secs_f64() / (num_runs as f64);
        let mean_time = time / (num_queries as f64);

        log::info!("knn-search time: {:.2e} seconds per query ...", mean_time);
    }
    log::info!("Moving on ...");
    log::info!("");

    Ok(())
}

fn main() -> Result<(), String> {
    env_logger::Builder::new().parse_filters("info").init();
    let mut results = vec![];

    for metric_name in ["euclidean", "cosine"] {
        if metric_name != "euclidean" {
            continue;
        }

        for &data_name in readers::ANOMALY_DATASETS {
            if data_name != "mnist" {
                continue;
            }

            results.push(search(data_name, metric_name, 1));
        }
    }

    println!(
        "Successful for {}/{} datasets.",
        results.iter().filter(|v| v.is_ok()).count(),
        results.len()
    );
    let failures = results.iter().filter(|v| v.is_err()).cloned().collect::<Vec<_>>();
    if !failures.is_empty() {
        println!("Failed for {}/{} datasets.", failures.len(), results.len());
        failures.into_iter().for_each(|v| println!("{:?}\n", v));
    }
    Ok(())
}
