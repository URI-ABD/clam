#![allow(missing_docs, dead_code)]

use std::hint::black_box;

use criterion::{measurement, *};

use rand::prelude::*;
use symagen::random_data;

use distances::{
    Number,
    vectors::{cosine, euclidean, l3_norm, l4_norm, manhattan},
};

fn cosine_f32(x: &[f32], y: &[f32]) -> f32 {
    let [xx, yy, xy] = x
        .iter()
        .zip(y.iter())
        .fold([0.0_f32; 3], |[xx, yy, xy], (&a, &b)| [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]);
    if xx == 0.0 || yy == 0.0 { 1.0 } else { 1.0 - xy * (xx * yy).sqrt().recip() }
}

fn euclidean_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn l3_norm_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs().powi(3)).sum::<f32>().powf(1.0 / 3.0)
}

fn l4_norm_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs().powi(4)).sum::<f32>().powf(1.0 / 4.0)
}

fn manhattan_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f32>()
}

fn bench_one<'a, T: Number, U: Number>(group: &mut BenchmarkGroup<'a, measurement::WallTime>, id: BenchmarkId, x: &[T], y: &[T], metric: fn(&[T], &[T]) -> U) {
    group.bench_with_input(id, &x.len(), |b, _| b.iter_with_large_drop(|| black_box(metric(x, y))));
}

fn big_f32(c: &mut Criterion) {
    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    let mut group = c.benchmark_group("VectorsF32");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    #[allow(clippy::type_complexity)]
    let metrics: &[(&str, fn(&[f32], &[f32]) -> f32, fn(&[f32], &[f32]) -> f32)] = &[
        ("L1", manhattan, manhattan_f32),
        ("L2", euclidean, euclidean_f32),
        ("L3", l3_norm, l3_norm_f32),
        ("L4", l4_norm, l4_norm_f32),
        ("Cosine", cosine, cosine_f32),
    ];

    for d in 3..=7 {
        let dimensionality = 10_u32.pow(d) as usize;
        let data = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rand::rngs::StdRng::seed_from_u64(d as u64));

        for &(name, metric, metric_f32) in metrics {
            let name_gen = format!("{}-generic", name);
            let id = BenchmarkId::new(name_gen, dimensionality);
            bench_one(&mut group, id, &data[0], &data[1], metric);

            let name_f32 = format!("{}-f32", name);
            let id = BenchmarkId::new(name_f32, dimensionality);
            bench_one(&mut group, id, &data[0], &data[1], metric_f32);
        }
    }
    group.finish();
}

criterion_group!(benches, big_f32);
criterion_main!(benches);
