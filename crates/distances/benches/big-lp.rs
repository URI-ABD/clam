#![allow(dead_code)]

use criterion::{measurement, *};
use rand::prelude::*;
use symagen::random_data;

use distances::{
    number::Float,
    vectors::{euclidean, l3_norm, l4_norm, manhattan, minkowski},
    Number,
};

fn bench_one<T: Number, U: Float>(
    group: &mut BenchmarkGroup<'_, measurement::WallTime>,
    p: i32,
    x: &[T],
    y: &[T],
    metric: impl Fn(&[T], &[T]) -> U,
) {
    let dimensionality = x.len();

    let id = BenchmarkId::new(format!("L{p}_con"), dimensionality);
    group.bench_with_input(id, &x.len(), |b, _| {
        b.iter_with_large_drop(|| black_box(metric(x, y)))
    });

    let id = BenchmarkId::new(format!("L{p}_gen"), dimensionality);
    let metric = minkowski::<T, U>(p);
    group.bench_with_input(id, &x.len(), |b, _| {
        b.iter_with_large_drop(|| black_box(metric(x, y)))
    });
}

fn big_lp_norms(c: &mut Criterion) {
    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    let mut group = c.benchmark_group("VectorsF32");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    #[allow(clippy::type_complexity)]
    let metrics: &[fn(&[f32], &[f32]) -> f32] = &[manhattan, euclidean, l3_norm, l4_norm];

    for d in 2..=7 {
        let dimensionality = 10_u32.pow(d) as usize;
        let data = random_data::random_tabular(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            &mut rand::rngs::StdRng::seed_from_u64(d as u64),
        );

        for (i, &metric) in metrics.iter().enumerate() {
            let p = i as i32 + 1;
            bench_one(&mut group, p, &data[0], &data[1], metric);
        }
    }
    group.finish();
}

criterion_group!(benches, big_lp_norms);
criterion_main!(benches);
