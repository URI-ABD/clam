use criterion::*;
use symagen::random_data;

use distances::strings::needleman_wunsch;

fn bench_with_edits(c: &mut Criterion) {
    let mut group = c.benchmark_group("NW-with-edits");
    // group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    let seed = 42;
    let cardinality = 10;
    let seq_len = 100;
    let alphabet = [
        "ACGT",                             // DNA
        "ACGTactg",                         // DNA with lowercase
        "ACGTURYSWKMBDHVN",                 // DNA with IUPAC
        "ACGTURYSWKMBDHVNacgturyswkmbdhvn", // DNA with IUPAC and lowercase
    ];

    group.throughput(Throughput::Elements((cardinality * cardinality) as u64));

    for len in [10, 25, 50, 100, 250, 500, 1000] {
        let sequences = random_data::random_string(cardinality, len, len, alphabet[0], seed);

        let id = BenchmarkId::new("distance-len", len);
        group.bench_with_input(id, &len, |b, _| {
            b.iter_with_large_drop(|| {
                black_box({
                    sequences
                        .iter()
                        .map(|x| {
                            sequences
                                .iter()
                                .map(|y| needleman_wunsch::nw_distance::<u32>(x, y))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
        });

        let id = BenchmarkId::new("recursive-edits-len", len);
        group.bench_with_input(id, &len, |b, _| {
            b.iter_with_large_drop(|| {
                black_box({
                    sequences
                        .iter()
                        .map(|x| {
                            sequences
                                .iter()
                                .map(|y| needleman_wunsch::edits_recursive::<u32>(x, y))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
        });

        let id = BenchmarkId::new("iterative-edits-len", len);
        group.bench_with_input(id, &len, |b, _| {
            b.iter_with_large_drop(|| {
                black_box({
                    sequences
                        .iter()
                        .map(|x| {
                            sequences
                                .iter()
                                .map(|y| needleman_wunsch::edits_iterative::<u32>(x, y))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
        });
    }

    for alf in alphabet {
        let sequences = random_data::random_string(cardinality, seq_len, seq_len, alf, seed);

        let id = BenchmarkId::new("distance-alf", alf.len());
        group.bench_with_input(id, &alf.len(), |b, _| {
            b.iter_with_large_drop(|| {
                black_box({
                    sequences
                        .iter()
                        .map(|x| {
                            sequences
                                .iter()
                                .map(|y| needleman_wunsch::nw_distance::<u32>(x, y))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
        });

        let id = BenchmarkId::new("recursive-edits-alf", alf.len());
        group.bench_with_input(id, &alf.len(), |b, _| {
            b.iter_with_large_drop(|| {
                black_box({
                    sequences
                        .iter()
                        .map(|x| {
                            sequences
                                .iter()
                                .map(|y| needleman_wunsch::edits_recursive::<u32>(x, y))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
        });

        let id = BenchmarkId::new("iterative-edits-alf", alf.len());
        group.bench_with_input(id, &alf.len(), |b, _| {
            b.iter_with_large_drop(|| {
                black_box({
                    sequences
                        .iter()
                        .map(|x| {
                            sequences
                                .iter()
                                .map(|y| needleman_wunsch::edits_iterative::<u32>(x, y))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_with_edits,);
criterion_main!(benches);
