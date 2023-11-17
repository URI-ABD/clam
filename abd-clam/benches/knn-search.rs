use criterion::*;

use symagen::random_data;

use abd_clam::{knn, Cakes, PartitionCriteria, VecDataset};

fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::vectors::euclidean(x, y)
}

fn euclidean_simd(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(x, y)
}

const METRICS: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] =
    &[("euclidean", euclidean), ("euclidean_simd", euclidean_simd)];

fn cakes(c: &mut Criterion) {
    let seed = 42;
    let (cardinality, dimensionality) = (100_000, 10);
    let (min_val, max_val) = (-1., 1.);

    let data = random_data::random_tabular_seedable::<f32>(cardinality, dimensionality, min_val, max_val, seed);

    let query = vec![0.0; dimensionality];

    for &(metric_name, metric) in METRICS {
        let mut group = c.benchmark_group(format!("knn-{metric_name}"));
        group
            // .sample_size(100)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(1))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        let dataset = VecDataset::<_, _, bool>::new("knn".to_string(), data.clone(), metric, false, None);
        let criteria = PartitionCriteria::default();
        let cakes = Cakes::new(dataset, Some(seed), &criteria);

        for k in (0..3).map(|v| 10_usize.pow(v)) {
            for &variant in knn::Algorithm::variants() {
                let id = BenchmarkId::new(variant.name(), k);
                group.bench_with_input(id, &k, |b, _| {
                    b.iter_with_large_drop(|| cakes.knn_search(&query, k, variant));
                });
            }

            let id = BenchmarkId::new("Linear", k);
            group.bench_with_input(id, &k, |b, _| {
                b.iter_with_large_drop(|| cakes.knn_search(&query, k, knn::Algorithm::Linear));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
