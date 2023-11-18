use criterion::*;
use symagen::random_data;

use distances::strings::levenshtein;

fn big_levenshtein(c: &mut Criterion) {
    let mut group = c.benchmark_group("Strings");

    for d in 0..=6 {
        let len = 100 * 2_usize.pow(d);
        let vecs = random_data::random_string(2, len, len, "ATCG", 42);
        let (x, y) = (&vecs[0], &vecs[1]);

        let id = BenchmarkId::new("Levenshtein", len);
        group.bench_with_input(id, &len, |b, _| {
            b.iter(|| black_box(levenshtein::<u16>(x, y)))
        });
    }
    group.finish();
}

criterion_group!(benches, big_levenshtein);
criterion_main!(benches);
