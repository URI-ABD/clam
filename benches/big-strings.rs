use criterion::*;
use symagen::random_data;

use distances::strings::levenshtein;

fn big_levenshtein(c: &mut Criterion) {
    let mut group = c.benchmark_group("Levenshtein");

    for d in 1..=5 {
        let len = d * 1_000;
        let vecs = random_data::random_string(2, len, len, "ATCG", 42);
        let (x, y) = (&vecs[0], &vecs[1]);

        let id = BenchmarkId::new("char", len);
        group.bench_with_input(id, &len, |b, _| {
            b.iter(|| black_box(|| levenshtein::<u16>(x, y)))
        });
    }
    group.finish();
}

criterion_group!(benches, big_levenshtein);
criterion_main!(benches);
