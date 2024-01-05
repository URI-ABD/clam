use criterion::*;
use rand::prelude::*;
use symagen::random_data;

use distances::number::Float;

/// Generates a vec of 10 million random non-negative f32s
fn gen_f32s() -> Vec<Vec<f32>> {
    random_data::random_tabular(
        1,
        10_000_000,
        0.0,
        1e10,
        &mut rand::rngs::StdRng::seed_from_u64(42),
    )
}

/// Generates a vec of 10 million random non-negative f64s
fn gen_f64s() -> Vec<Vec<f64>> {
    random_data::random_tabular(
        1,
        10_000_000,
        0.0,
        1e10,
        &mut rand::rngs::StdRng::seed_from_u64(42),
    )
}

fn inv_sqrt_f32(c: &mut Criterion) {
    let floats = gen_f32s();
    let floats = floats.first().unwrap();

    let mut group = c.benchmark_group("InvSqrtF32");
    group.throughput(Throughput::Elements(floats.len() as u64));

    group.bench_function("from_core", |b| {
        b.iter_with_large_drop(|| {
            black_box(floats.iter().map(|&x| x.sqrt().recip()).collect::<Vec<_>>())
        })
    });

    group.bench_function("quake_iii", |b| {
        b.iter(|| black_box(floats.iter().map(|&x| x.inv_sqrt()).collect::<Vec<_>>()))
    });
    group.finish();
}

fn inv_sqrt_f64(c: &mut Criterion) {
    let floats = gen_f64s();
    let floats = floats.first().unwrap();

    let mut group = c.benchmark_group("InvSqrtF64");
    group.throughput(Throughput::Elements(floats.len() as u64));

    group.bench_function("from_core", |b| {
        b.iter_with_large_drop(|| {
            black_box(floats.iter().map(|&x| x.sqrt().recip()).collect::<Vec<_>>())
        })
    });

    group.bench_function("quake_iii", |b| {
        b.iter(|| black_box(floats.iter().map(|&x| x.inv_sqrt()).collect::<Vec<_>>()))
    });
    group.finish();
}

criterion_group!(benches, inv_sqrt_f32, inv_sqrt_f64);
criterion_main!(benches);
