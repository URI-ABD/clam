use criterion::*;

use clam::core::cluster::Cluster;
use clam::core::cluster_criteria::PartitionCriteria;
use clam::core::dataset::VecVec;

pub mod utils;

use utils::distances;

fn partition_car(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition-car");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let (data, _, name) = utils::make_data(n, 100, 0);
        let data = VecVec::new(data, distances::euclidean, name, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        if n < 250 {
            let id = BenchmarkId::new("100-euclidean", n);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| Cluster::new_root(&data).build().partition(&criteria, true));
            });
        }

        let id = BenchmarkId::new("par-100-euclidean", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).build().par_partition(&criteria, true));
        });
    }
    group.finish();
}

fn partition_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition-dim");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let (data, _, name) = utils::make_data(100, n, 0);
        let data = VecVec::new(data, distances::euclidean, name, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        if n < 250 {
            let id = BenchmarkId::new("100k-euclidean", n);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| Cluster::new_root(&data).build().partition(&criteria, true));
            });
        }

        let id = BenchmarkId::new("par-100k-euclidean", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).build().par_partition(&criteria, true));
        });
    }
    group.finish();
}

fn partition_met(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition-met");
    group.significance_level(0.025).sample_size(10);

    group.sampling_mode(SamplingMode::Flat);

    for (metric_name, metric) in distances::METRICS {
        let (data, _, name) = utils::make_data(100, 100, 0);
        let data = VecVec::new(data, metric, name, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        let id = BenchmarkId::new("100k-100", metric_name);
        group.bench_with_input(id, &metric, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).build().partition(&criteria, true));
        });

        let id = BenchmarkId::new("par-100k-100", metric_name);
        group.bench_with_input(id, &metric, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).build().par_partition(&criteria, true));
        });
    }
    group.finish();
}

criterion_group!(benches, partition_car, partition_dim, partition_met); // partition_car, partition_dim, partition_met
criterion_main!(benches);
