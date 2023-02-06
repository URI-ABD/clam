use clam::prelude::*;

mod readers;

#[allow(clippy::single_element_loop)]
fn search(data_name: &str, metric_name: &str, num_runs: usize) -> Result<(), String> {
    log::info!("");

    let (features, _) = readers::read_anomaly_data(data_name, true)?;
    let dataset = clam::Tabular::new(&features, data_name.to_string());
    let run_name = format!("{data_name}__{metric_name}");
    let shape_name = format!("({}, {})", dataset.cardinality(), dataset.dimensionality());

    let num_queries = 1000;
    let queries = (0..num_queries)
        .map(|i| dataset.get(i % dataset.cardinality()))
        .collect::<Vec<_>>();
    let num_queries = queries.len();
    log::info!("Running {run_name} data with shape {shape_name} and {num_queries} queries ...");

    let metric = metric_from_name::<f32>(metric_name, false).unwrap();
    let space = clam::TabularSpace::new(&dataset, metric.as_ref());
    let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1);

    let start = std::time::Instant::now();
    let cakes = clam::CAKES::new(&space).build(&partition_criteria);
    let diameter = cakes.diameter();
    let build_time = start.elapsed().as_secs_f64();
    log::info!(
        "Built tree to a depth of {} in {build_time:.2e} seconds ...",
        cakes.depth()
    );

    // for k in [1, 10, 100] {
    for k in [100] {
        if k > dataset.cardinality() {
            continue;
        }
        let r = diameter / k as f64;
        log::info!("Using radius: {r:.12} ...");

        let queries_radii = queries.iter().map(|&q| (q, r)).collect::<Vec<_>>();

        let start = std::time::Instant::now();
        (0..num_runs).for_each(|_| {
            cakes.batch_rnn_search(&queries_radii);
        });
        let time = start.elapsed().as_secs_f64() / (num_runs as f64);
        let mean_time = time / (num_queries as f64);

        log::info!("knn-search time: {mean_time:.2e} seconds per query for {num_queries} queries ...");
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
            if data_name != "cover" {
                continue;
            }

            results.push(search(data_name, metric_name, 10));
        }
    }

    println!(
        "Successful for {}/{} datasets.",
        results.iter().filter(|v| v.is_ok()).count(),
        results.len()
    );
    let failures = results.iter().filter(|v| v.is_err()).cloned().collect::<Vec<_>>();
    if !failures.is_empty() {
        log::info!("Failed for {}/{} datasets.", failures.len(), results.len());
        failures.into_iter().for_each(|v| log::info!("{v:?}\n"));
    }
    Ok(())
}
