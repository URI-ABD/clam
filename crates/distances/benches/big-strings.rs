use criterion::*;
use symagen::random_data;

use distances::strings::levenshtein;

fn big_levenshtein(c: &mut Criterion) {
    let mut group = c.benchmark_group("Levenshtein");

    for d in 2..=4 {
        let len = 10_usize.pow(d);
        let vecs = random_data::random_string(2, len, len, "ATCGN", 42);
        let (x, y) = (&vecs[0], &vecs[1]);

        let id = BenchmarkId::new("Distances", len);
        group.bench_with_input(id, &len, |b, _| b.iter(|| black_box(levenshtein::<u16>(x, y))));

        let id = BenchmarkId::new("StringZilla", len);
        group.bench_with_input(id, &len, |b, _| b.iter(|| black_box(sz_lev(x, y))));
    }
    group.finish();
}

/// Use the StringZilla implementation of the Levenshtein distance.
fn sz_lev(x: &str, y: &str) -> u16 {
    stringzilla::sz::edit_distance(x, y) as u16
}

criterion_group!(benches, big_levenshtein);
criterion_main!(benches);
