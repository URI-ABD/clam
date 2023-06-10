use criterion::*;

use clam::distances::alignment_helpers;
use clam::distances::strings;
use clam::utils::helpers;

fn needleman_wunsch_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("needleman-wunsch-dim");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let data = helpers::gen_data_u8(2, n, 65, 68, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| strings::needleman_wunsch_with_edits::<u8, u32>(x, y));
        });
    }

    group.finish();
}

fn needleman_wunsch_alphabet_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("needleman-wunsch-alphabet-size");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for alphabet_size in [4, 8, 16, 32] {
        let data = helpers::gen_data_u8(2, 50, 65, 65 + alphabet_size, 42);
        let x = &data[0];
        let y = &data[1];
        let id = BenchmarkId::new("nw", alphabet_size);
        group.bench_with_input(id, &alphabet_size, |b, _| {
            b.iter_with_large_drop(|| strings::needleman_wunsch_with_edits::<u8, u32>(x, y));
        });
    }

    group.finish();
}

fn traceback_iterative_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("traceback-iterative-dim");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let data = helpers::gen_data_u8(2, n, 65, 68, 42);
        let x = &data[0];
        let y = &data[1];
        let table: Vec<Vec<(usize, alignment_helpers::Direction)>> = alignment_helpers::compute_nw_table(x, y);
        let id = BenchmarkId::new("nw-traceback-iterative", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| alignment_helpers::traceback_iterative(&table, (x, y)));
        });
    }

    group.finish();
}

fn traceback_recursive_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("traceback-recursive-dim");
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    for n in [10, 25, 50, 100, 250, 500, 1000] {
        let data = helpers::gen_data_u8(2, n, 65, 68, 42);
        let x = &data[0];
        let y = &data[1];
        let table: Vec<Vec<(usize, alignment_helpers::Direction)>> = alignment_helpers::compute_nw_table(x, y);
        let id = BenchmarkId::new("nw-traceback-recursive", n);
        group.bench_with_input(id, &n, |b, _| {
            b.iter_with_large_drop(|| alignment_helpers::traceback_recursive(&table, (x, y)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    needleman_wunsch_dim,
    needleman_wunsch_alphabet_size,
    traceback_recursive_dim,
    traceback_iterative_dim
);
criterion_main!(benches);
