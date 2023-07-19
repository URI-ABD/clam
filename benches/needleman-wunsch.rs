use criterion::*;

use symagen::random_data;

use distances::strings::needleman_wunsch;

fn needleman_wunsch_recursive_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("needleman-wunsch-recursive-dim");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let data = random_data::random_string(2, n, n, "ATCG", 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| needleman_wunsch::with_edits_recursive::<u32>(x, y));
        });
    }

    group.finish();
}

fn needleman_wunsch_iterative_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("needleman-wunsch-iterative-dim");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let data = random_data::random_string(2, n, n, "ATCG", 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| needleman_wunsch::with_edits_iterative::<u32>(x, y));
        });
    }

    group.finish();
}

fn needleman_wunsch_recursive_alphabet_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("needleman-wunsch-recursive-alphabet-size");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    let all_alphabet = [
        "ACGT",                             // DNA
        "ACGTactg",                         // DNA with lowercase
        "ACGTURYSWKMBDHVN",                 // DNA with IUPAC
        "ACGTURYSWKMBDHVNacgturyswkmbdhvn", // DNA with IUPAC and lowercase
    ];

    for alphabet in all_alphabet {
        let data = random_data::random_string(2, 100, 100, alphabet, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", alphabet.len());
        group.bench_with_input(id, &alphabet.len(), |b, _| {
            b.iter_with_large_drop(|| needleman_wunsch::with_edits_recursive::<u32>(x, y));
        });
    }

    group.finish();
}

fn needleman_wunsch_iterative_alphabet_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("needleman-wunsch-iterative-alphabet-size");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    let all_alphabet = [
        "ACGT",                             // DNA
        "ACGTactg",                         // DNA with lowercase
        "ACGTURYSWKMBDHVN",                 // DNA with IUPAC
        "ACGTURYSWKMBDHVNacgturyswkmbdhvn", // DNA with IUPAC and lowercase
    ];

    for alphabet in all_alphabet {
        let data = random_data::random_string(2, 100, 100, alphabet, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", alphabet.len());
        group.bench_with_input(id, &alphabet.len(), |b, _| {
            b.iter_with_large_drop(|| needleman_wunsch::with_edits_recursive::<u32>(x, y));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    needleman_wunsch_recursive_dim,
    needleman_wunsch_iterative_dim,
    needleman_wunsch_recursive_alphabet_size,
    needleman_wunsch_iterative_alphabet_size
);
criterion_main!(benches);
