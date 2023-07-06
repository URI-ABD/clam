use criterion::*;
use rand::prelude::*;

use distances::number::Float;

/// Generates a vec of 10 million random non-negative f32s
fn gen_f32s() -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut vec = Vec::with_capacity(10_000_000);
    for _ in 0..10_000_000 {
        vec.push(rng.gen::<f32>().abs());
    }
    vec
}

/// Generates a vec of 10 million random non-negative f64s
fn gen_f64s() -> Vec<f64> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut vec = Vec::with_capacity(10_000_000);
    for _ in 0..10_000_000 {
        vec.push(rng.gen::<f64>().abs());
    }
    vec
}

fn inv_sqrt_f32(c: &mut Criterion) {
    let floats = gen_f32s();

    let mut group = c.benchmark_group("InvSqrtF32");
    group.throughput(Throughput::Elements(floats.len() as u64));

    group.bench_function("from_core", |b| {
        b.iter_with_large_drop(|| {
            black_box(|| floats.iter().map(|&x| x.sqrt().recip()).collect::<Vec<_>>())
        })
    });

    group.bench_function("quake_iii", |b| {
        b.iter(|| black_box(|| floats.iter().map(|&x| x.inv_sqrt()).collect::<Vec<_>>()))
    });
    group.finish();
}

fn inv_sqrt_f64(c: &mut Criterion) {
    let floats = gen_f64s();

    let mut group = c.benchmark_group("InvSqrtF64");
    group.throughput(Throughput::Elements(floats.len() as u64));

    group.bench_function("from_core", |b| {
        b.iter_with_large_drop(|| {
            black_box(|| floats.iter().map(|&x| x.sqrt().recip()).collect::<Vec<_>>())
        })
    });

    group.bench_function("quake_iii", |b| {
        b.iter(|| black_box(|| floats.iter().map(|&x| x.inv_sqrt()).collect::<Vec<_>>()))
    });
    group.finish();
}

criterion_group!(benches, inv_sqrt_f32, inv_sqrt_f64);
criterion_main!(benches);
