use abd_clam::{
    cakes::{self, Algorithm, OffsetBall},
    partition::ParPartition,
    Ball, Cluster, FlatVec, Metric,
};
use criterion::*;
use rand::prelude::*;

const METRICS: &[(&str, fn(&String, &String) -> u16)] = &[
    ("levenshtein", |x: &String, y: &String| {
        distances::strings::levenshtein(x, y)
    }),
    ("needleman-wunsch", |x: &String, y: &String| {
        distances::strings::nw_distance(x, y)
    }),
];

fn genomic_search(c: &mut Criterion) {
    let seed_length = 100;
    let alphabet = "ACTGN".chars().collect::<Vec<_>>();
    let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
    let penalties = distances::strings::Penalties::default();
    let num_clumps = 256;
    let clump_size = 32;
    let clump_radius = 3_u16;
    let (_, genomes) = symagen::random_edits::generate_clumped_data(
        &seed_string,
        penalties,
        &alphabet,
        num_clumps,
        clump_size,
        clump_radius,
    )
    .into_iter()
    .unzip::<_, _, Vec<_>, Vec<_>>();

    let seed = 42;
    let num_queries = 10;
    let queries = {
        let mut indices = (0..genomes.len()).collect::<Vec<_>>();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        indices
            .into_iter()
            .take(num_queries)
            .map(|i| genomes[i].clone())
            .collect::<Vec<_>>()
    };

    let seed = Some(seed);
    let radii = vec![2, 3, 4, 6, 16];
    let ks = vec![1, 10, 20];
    for &(metric_name, distance_fn) in METRICS {
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(genomes.clone(), metric).unwrap();

        let criteria = |c: &Ball<_>| c.cardinality() > 1;
        let root = Ball::par_new_tree(&data, &criteria, seed);
        bench_on_root(c, false, metric_name, &root, &data, &queries, &radii, &ks);

        let mut data = data;
        let root = OffsetBall::from_ball_tree(root, &mut data);
        bench_on_root(c, true, metric_name, &root, &data, &queries, &radii, &ks);
    }
}

fn bench_on_root<C>(
    c: &mut Criterion,
    permuted: bool,
    metric_name: &str,
    root: &C,
    data: &FlatVec<String, u16, usize>,
    queries: &[String],
    radii: &[u16],
    ks: &[usize],
) where
    C: Cluster<u16> + cakes::cluster::ParSearchable<String, u16, FlatVec<String, u16, usize>>,
{
    let permuted = if permuted { "-permuted" } else { "" };

    let mut group = c.benchmark_group(format!("genomic-search-{}{}", metric_name, permuted));

    for &radius in radii {
        let id = BenchmarkId::new("RnnClustered", radius);
        group.bench_with_input(id, &radius, |b, &radius| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::RnnClustered(radius)));
        });
    }

    for &k in ks {
        let id = BenchmarkId::new("KnnRepeatedRnn", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::KnnRepeatedRnn(k, 2)));
        });

        let id = BenchmarkId::new("KnnBreadthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::KnnBreadthFirst(k)));
        });

        let id = BenchmarkId::new("KnnDepthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::KnnDepthFirst(k)));
        });
    }

    for &radius in radii {
        let id = BenchmarkId::new("ParRnnClustered", radius);
        group.bench_with_input(id, &radius, |b, &radius| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, queries, Algorithm::RnnClustered(radius)));
        });
    }

    for &k in ks {
        let id = BenchmarkId::new("ParKnnRepeatedRnn", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, &queries, Algorithm::KnnRepeatedRnn(k, 2)));
        });

        let id = BenchmarkId::new("ParKnnBreadthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, &queries, Algorithm::KnnBreadthFirst(k)));
        });

        let id = BenchmarkId::new("ParKnnDepthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, &queries, Algorithm::KnnDepthFirst(k)));
        });
    }
}

criterion_group!(benches, genomic_search);
criterion_main!(benches);
