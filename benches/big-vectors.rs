use criterion::*;
use symagen::random_data;

use distances::vectors::{cosine, euclidean, euclidean_sq, l3_norm, l4_norm, manhattan};

fn l1_norm(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
}

fn l2_norm(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn sq_l2_norm(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn big_minkowski(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vectors");

    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    for d in 0..=5 {
        let dimensionality = 1_000 * 2_u32.pow(d) as usize;
        let vecs = random_data::random_f32(cardinality, dimensionality, min_val, max_val, d as u64);

        let id = BenchmarkId::new("L1", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(manhattan(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L1-mono", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(l1_norm(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(euclidean::<f32, f32>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("L2-mono", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(l2_norm(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("SQ_L2", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(euclidean_sq::<f32, f32>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("SQ_L2-mono", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(sq_l2_norm(&vecs[0], &vecs[1])))
        });

        if d < 4 {
            let id = BenchmarkId::new("L3", dimensionality);
            group.bench_with_input(id, &dimensionality, |b, _| {
                b.iter(|| black_box(l3_norm::<f32, f32>(&vecs[0], &vecs[1])))
            });

            let id = BenchmarkId::new("L4", dimensionality);
            group.bench_with_input(id, &dimensionality, |b, _| {
                b.iter(|| black_box(l4_norm::<f32, f32>(&vecs[0], &vecs[1])))
            });
        }

        let id = BenchmarkId::new("Cosine", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(cosine::<f32, f32>(&vecs[0], &vecs[1])))
        });
    }
    group.finish();
}

criterion_group!(benches, big_minkowski);
criterion_main!(benches);
