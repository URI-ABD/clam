#![allow(dead_code)]

use criterion::{measurement, *};

use rand::prelude::*;
use symagen::random_data;

use distances::{
    vectors::{cosine, euclidean, l3_norm, l4_norm, manhattan},
    Number,
};

fn bench_one<'a, T: Number, U: Number>(
    group: &mut BenchmarkGroup<'a, measurement::WallTime>,
    id: BenchmarkId,
    x: &[T],
    y: &[T],
    metric: fn(&[T], &[T]) -> U,
) {
    group.bench_with_input(id, &x.len(), |b, _| {
        b.iter_with_large_drop(|| black_box(metric(x, y)))
    });
}

fn big_f32(c: &mut Criterion) {
    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    let mut group = c.benchmark_group("VectorsF32");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    #[allow(clippy::type_complexity)]
    let metrics: &[(&str, fn(&[f32], &[f32]) -> f32)] = &[
        ("L1", manhattan),
        ("L2", euclidean),
        ("L3", l3_norm),
        ("L4", l4_norm),
        ("Cosine", cosine),
    ];

    for d in 2..=7 {
        let dimensionality = 10_u32.pow(d) as usize;
        let data = random_data::random_tabular(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            &mut rand::rngs::StdRng::seed_from_u64(d as u64),
        );

        for &(name, metric) in metrics {
            let id = BenchmarkId::new(name, dimensionality);
            bench_one(&mut group, id, &data[0], &data[1], metric);
        }
    }
    group.finish();
}

fn big_u32(c: &mut Criterion) {
    let (cardinality, min_val, max_val) = (2, 0, 10_000);

    let mut group = c.benchmark_group("VectorsU32");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    #[allow(clippy::type_complexity)]
    let metrics: &[(&str, fn(&[u32], &[u32]) -> f32)] = &[
        ("L2", euclidean),
        ("L3", l3_norm),
        ("L4", l4_norm),
        ("Cosine", cosine),
    ];

    for d in 2..=7 {
        let dimensionality = 10_u32.pow(d) as usize;
        let data = random_data::random_tabular(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            &mut rand::rngs::StdRng::seed_from_u64(d as u64),
        );

        let id = BenchmarkId::new("L1", dimensionality);
        bench_one(&mut group, id, &data[0], &data[1], manhattan);

        for &(name, metric) in metrics {
            let id = BenchmarkId::new(name, dimensionality);
            bench_one(&mut group, id, &data[0], &data[1], metric);
        }
    }
    group.finish();
}

criterion_group!(benches, big_f32);
criterion_main!(benches);
