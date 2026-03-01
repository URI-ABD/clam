#![expect(missing_docs, clippy::missing_docs_in_private_items, clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use stringzilla::szs::{DeviceScope, LevenshteinDistances};

use distances::strings::levenshtein;

fn big_levenshtein(c: &mut Criterion) {
    /// Use the `StringZilla` implementation of the Levenshtein distance.
    fn sz_lev_builder() -> impl Fn(&str, &str) -> u16 {
        let device = DeviceScope::default().unwrap();
        let szla_engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
        move |x, y| szla_engine.compute(&device, &[x], &[y]).unwrap()[0] as u16
    }
    let sz_lev = sz_lev_builder();

    let mut group = c.benchmark_group("Levenshtein");
    for d in 2..=4 {
        let len = 10_usize.pow(d);
        let mut rng = StdRng::seed_from_u64(42);
        let vecs = symagen::random_data::random_string(2, len, len, "ATCGN", &mut rng);
        let (x, y) = (&vecs[0], &vecs[1]);

        let id = BenchmarkId::new("Distances", len);
        group.bench_with_input(id, &len, |b, _| b.iter(|| black_box(levenshtein::<u16>(x, y))));

        let id = BenchmarkId::new("StringZilla", len);
        group.bench_with_input(id, &len, |b, _| b.iter(|| black_box(sz_lev(x, y))));
    }
    group.finish();
}

criterion_group!(benches, big_levenshtein);
criterion_main!(benches);
