pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use clam::prelude::*;

use utils::anomaly_readers;

fn partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Partition");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0));
        .sample_size(10);

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();
        // if features.len() > 50_000 {
        //     continue;
        // }

        let dataset = clam::Tabular::new(&features, data_name.to_string());
        // let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let euclidean = clam::metric::Euclidean { is_expensive: false };
        let space = clam::TabularSpace::new(&dataset, &euclidean);
        let partition_criteria =
            clam::PartitionCriteria::<f32, clam::TabularSpace<f32>>::new(true).with_min_cardinality(1);

        let cardinality = dataset.cardinality();
        let dimensionality = dataset.dimensionality();
        println!("\nMaking tree on {data_name} data with {cardinality} cardinality and {dimensionality}");
        let root = Cluster::new_root(&space).partition(&partition_criteria, true);
        let subtree = root.subtree();
        let num_clusters = subtree.len();
        let max_leaf_depth = subtree.iter().map(|c| c.depth()).max().unwrap();
        println!("Got a tree of depth {max_leaf_depth} with {num_clusters} total clusters.\n");

        group.bench_function(data_name, |b| {
            b.iter_with_large_drop(|| Cluster::new_root(&space).partition(&partition_criteria, true))
        });
    }

    group.finish();
}

criterion_group!(benches, partition);
criterion_main!(benches);
