#![allow(missing_docs)]

use std::hint::black_box;

use criterion::*;
use rand::prelude::*;
use symagen::random_data;

use distances::blas;
use distances::simd;

use distances::vectors::{dot_product as dot_generic, euclidean as l2_generic, euclidean_sq as l2_sq_generic};

fn simd_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdF32");

    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    for d in 0..=5 {
        let dimensionality = 1_000 * 2_u32.pow(d) as usize;
        let vecs = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rand::rngs::StdRng::seed_from_u64(d as u64));

        let id = BenchmarkId::new("L2-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(l2_generic::<_, f32>(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(simd::euclidean_f32(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-blas", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(blas::euclidean_f32(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-sq-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(l2_sq_generic::<_, f32>(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-sq-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(simd::euclidean_sq_f32(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-sq-blas", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(blas::euclidean_sq_f32(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("Dot-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(dot_generic(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("Dot-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(simd::dot_product_f32(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("Dot-blas", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(blas::dot_f32(&vecs[0], &vecs[1]))));
    }
    group.finish();
}

fn simd_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdF64");

    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    for d in 0..=5 {
        let dimensionality = 1_000 * 2_u32.pow(d) as usize;
        let vecs = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rand::rngs::StdRng::seed_from_u64(d as u64));

        let id = BenchmarkId::new("L2-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(l2_generic::<_, f64>(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(simd::euclidean_f64(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-blas", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(blas::euclidean_f64(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-sq-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(l2_sq_generic::<_, f64>(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("L2-sq-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(simd::euclidean_sq_f64(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("Dot-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(dot_generic(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("Dot-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(simd::dot_product_f64(&vecs[0], &vecs[1]))));

        let id = BenchmarkId::new("Dot-blas", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| b.iter(|| black_box(blas::dot_f64(&vecs[0], &vecs[1]))));
    }
    group.finish();
}

criterion_group!(benches, simd_f32, simd_f64);
criterion_main!(benches);
