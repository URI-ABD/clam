use itertools::Itertools;

use clam::geometry::triangle;
use clam::prelude::*;

#[inline(never)]
fn check_triangle([ab, ac, bc]: [f64; 3]) -> bool {
    let abc = triangle::Triangle::with_edges_unchecked([ab, ac, bc]);
    abc.r_sq() > 0. && abc.cm_sq() > 0.
}

fn triangles(n_runs: usize) {
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

    let count = (0..n_runs)
        .map(|_| all_sides.iter().filter(|&&sides| check_triangle(sides)).count())
        .last()
        .unwrap();

    let rate = (10_000. * count.as_f64() / all_sides.len().as_f64()).floor() / 100.;
    let n_total = n_rows.pow(3);
    println!("Ran triangles {n_runs} times and was {rate:.2}% successful. There were {count} legal triangles among {n_total}.");
}

fn main() {
    triangles(10)
}
