use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use itertools::Itertools;

use clam::geometry::triangle;
use clam::prelude::*;

#[inline(never)]
fn check([ab, ac, bc]: [f64; 3]) -> bool {
    let abc = triangle::Triangle::with_edges_unchecked([ab, ac, bc]);
    abc.r_sq() > 0. && abc.cm_sq() > 0.
}

fn triangles(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triangle");
    group.significance_level(0.05).sample_size(100);

    let [n_rows, n_cols] = [200, 10];
    let data = clam::utils::helpers::gen_data(42, [n_rows, n_cols]);
    let data = data.iter().map(|row| &row[..]).collect::<Vec<_>>();
    let metric = clam::metric::Euclidean { is_expensive: false };
    let distances = metric.pairwise(&data);

    let all_sides = (0..n_rows)
        .cartesian_product(0..n_rows)
        .cartesian_product(0..n_rows)
        .into_iter()
        .filter_map(|((a, b), c)| {
            let ab = distances[a][b];
            let bc = distances[b][c];
            let ac = distances[a][c];
            let abc = triangle::Triangle::with_edges(['a', 'b', 'c'], [ab, ac, bc]);
            abc.ok().map(|_| [ab, ac, bc])
        })
        .collect::<Vec<_>>();

    let n_triangles = n_rows.pow(3);
    let bench_name = format!("{n_triangles}");
    group.bench_function(&bench_name, |b| {
        b.iter_with_large_drop(|| all_sides.iter().filter(|&&sides| check(sides)).collect::<Vec<_>>())
    });

    group.finish();
}

criterion_group!(benches, triangles);
criterion_main!(benches);
