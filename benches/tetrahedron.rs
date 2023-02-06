use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use itertools::Itertools;

use clam::geometry::tetrahedron;
use clam::prelude::*;

#[inline(never)]
fn check([ab, ac, bc, ad, bd, cd]: [f64; 6]) -> bool {
    let mut abcd = tetrahedron::Tetrahedron::with_edges_unchecked([ab, ac, bc, ad, bd, cd]);
    let od_sq = abcd.od_sq();
    // let od = abcd.od();
    !od_sq.is_nan() && !od_sq.is_infinite() && od_sq >= 0. // && od >= 0.
}

fn tetrahedra(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tetrahedron");
    group.significance_level(0.05).sample_size(100);

    let [n_rows, n_cols] = [50, 10];
    let data = clam::utils::helpers::gen_data(42, [n_rows, n_cols]);
    let data = data.iter().map(|row| &row[..]).collect::<Vec<_>>();
    let metric = clam::metric::Euclidean { is_expensive: false };
    let distances = metric.pairwise(&data);

    let all_sides = (0..n_rows)
        .cartesian_product(0..n_rows)
        .cartesian_product(0..n_rows)
        .cartesian_product(0..n_rows)
        .into_iter()
        .filter_map(|(((a, b), c), d)| {
            let edges = [
                distances[a][b],
                distances[b][c],
                distances[a][c],
                distances[a][d],
                distances[b][d],
                distances[c][d],
            ];
            let abcd = tetrahedron::Tetrahedron::with_edges(['a', 'b', 'c', 'd'], edges);
            abcd.ok().map(|_| edges)
        })
        .collect::<Vec<_>>();

    let n_tetrahedra = n_rows.pow(4);
    let bench_name = format!("{n_tetrahedra}");
    group.bench_function(&bench_name, |b| {
        b.iter_with_large_drop(|| all_sides.iter().filter(|&&sides| check(sides)).collect::<Vec<_>>())
    });

    group.finish();
}

criterion_group!(benches, tetrahedra);
criterion_main!(benches);
