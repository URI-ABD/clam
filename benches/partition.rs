use criterion::*;

use clam::core::cluster::Cluster;
use clam::core::cluster_criteria::PartitionCriteria;
use clam::core::dataset::VecVec;
use clam::distances;
use clam::utils::helpers;

fn partition_car(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition-car");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let data = helpers::gen_data_f32(n * 1_000, 10, 0., 1., 42);
        let name = format!("{n}k-10");

        let data = VecVec::new(data, distances::f32::euclidean, name, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        if n < 250 {
            let id = BenchmarkId::new("100-euclidean", n);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| Cluster::new_root(&data).with_seed(42).partition(&criteria, true));
            });
        }

        let id = BenchmarkId::new("par-100-euclidean", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).with_seed(42).par_partition(&criteria, true));
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
        let data = helpers::gen_data_f32(100_000, n, 0., 1., 42);
        let name = format!("100k-{n}");

        let data = VecVec::new(data, distances::f32::euclidean, name, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        if n < 250 {
            let id = BenchmarkId::new("100k-euclidean", n);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| Cluster::new_root(&data).with_seed(42).partition(&criteria, true));
            });
        }

        let id = BenchmarkId::new("par-100k-euclidean", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).with_seed(42).par_partition(&criteria, true));
        });
    }
    group.finish();
}

fn partition_met(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition-met");
    group.significance_level(0.025).sample_size(10);

    group.sampling_mode(SamplingMode::Flat);

    for (metric_name, metric) in distances::f32::METRICS {
        let data = helpers::gen_data_f32(100_000, 100, 0., 1., 42);
        let name = "100k-100".to_string();

        let data = VecVec::new(data, metric, name, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        let id = BenchmarkId::new("100k-100", metric_name);
        group.bench_with_input(id, &metric, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).with_seed(42).partition(&criteria, true));
        });

        let id = BenchmarkId::new("par-100k-100", metric_name);
        group.bench_with_input(id, &metric, |b, _| {
            b.iter_with_large_drop(|| Cluster::new_root(&data).with_seed(42).par_partition(&criteria, true));
        });
    }
    group.finish();
}

criterion_group!(benches, partition_car, partition_dim, partition_met); // partition_car, partition_dim, partition_met
criterion_main!(benches);
