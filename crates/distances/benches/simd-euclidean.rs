use criterion::*;
use rand::prelude::*;
use symagen::random_data;

use distances::simd;

use distances::vectors::{euclidean as l2_generic, euclidean_sq as l2_sq_generic};

fn simd_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdF32");

    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    for d in 0..=5 {
        let dimensionality = 1_000 * 2_u32.pow(d) as usize;
        let vecs = random_data::random_tabular(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            &mut rand::rngs::StdRng::seed_from_u64(d as u64),
        );

        let id = BenchmarkId::new("L2-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(l2_generic::<_, f32>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(simd::euclidean_f32(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2-sq-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(l2_sq_generic::<_, f32>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2-sq-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(simd::euclidean_sq_f32(&vecs[0], &vecs[1])))
        });
    }
    group.finish();
}

fn simd_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdF64");

    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    for d in 0..=5 {
        let dimensionality = 1_000 * 2_u32.pow(d) as usize;
        let vecs = random_data::random_tabular(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            &mut rand::rngs::StdRng::seed_from_u64(d as u64),
        );

        let id = BenchmarkId::new("L2-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(l2_generic::<_, f64>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(simd::euclidean_f64(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2-sq-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(l2_sq_generic::<_, f64>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2-sq-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(simd::euclidean_sq_f64(&vecs[0], &vecs[1])))
        });
    }
    group.finish();
}

criterion_group!(benches, simd_f32, simd_f64);
criterion_main!(benches);
