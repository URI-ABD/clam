use criterion::*;

use abd_clam::distances::strings;
use abd_clam::utils::synthetic_data;

fn needleman_wunsch_recursive_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("needleman-wunsch-recursive-dim");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let data = synthetic_data::random_u8(2, n, 65, 68, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| strings::needleman_wunsch_with_edits_recursive::<u8, u32>(x, y));
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
        let data = synthetic_data::random_u8(2, n, 65, 68, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| strings::needleman_wunsch_with_edits_iterative::<u8, u32>(x, y));
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

    for alphabet_size in [4, 8, 16, 32] {
        let data = synthetic_data::random_u8(2, 50, 65, 65 + alphabet_size, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", alphabet_size);
        group.bench_with_input(id, &alphabet_size, |b, _| {
            b.iter_with_large_drop(|| strings::needleman_wunsch_with_edits_recursive::<u8, u32>(x, y));
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

    for alphabet_size in [4, 8, 16, 32] {
        let data = synthetic_data::random_u8(2, 50, 65, 65 + alphabet_size, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", alphabet_size);
        group.bench_with_input(id, &alphabet_size, |b, _| {
            b.iter_with_large_drop(|| strings::needleman_wunsch_with_edits_recursive::<u8, u32>(x, y));
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
