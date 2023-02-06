use itertools::Itertools;

use clam::geometry::tetrahedron;
use clam::prelude::*;

#[inline(never)]
fn check_tetrahedron([ab, ac, bc, ad, bd, cd]: [f64; 6]) -> bool {
    let mut abcd = tetrahedron::Tetrahedron::with_edges_unchecked([ab, ac, bc, ad, bd, cd]);
    let od_sq = abcd.od_sq();
    !od_sq.is_nan() && !od_sq.is_infinite() && od_sq >= 0.
}

fn tetrahedra(n_runs: usize) {
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

    let count = (0..n_runs)
        .map(|_| all_sides.iter().filter(|&&sides| check_tetrahedron(sides)).count())
        .last()
        .unwrap();

    let rate = (10_000. * count.as_f64() / all_sides.len().as_f64()).floor() / 100.;
    let n_total = n_rows.pow(4);
    println!("Ran tetrahedra {n_runs} times and was {rate:.2}% successful. There were {count} legal tetrahedra among {n_total}.");
}

fn main() {
    tetrahedra(10)
}
