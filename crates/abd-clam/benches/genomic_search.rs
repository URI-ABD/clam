//! Benchmark for genomic search.

mod utils;

use abd_clam::{cakes::OffsetBall, partition::ParPartition, Ball, Cluster, FlatVec, Metric};
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
    let num_clumps = 128;
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
    let radii = vec![1, 2, 3, 4, 6, 10];
    let ks = vec![1, 10, 20];
    for &(metric_name, distance_fn) in METRICS {
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(genomes.clone(), metric).unwrap();

        let criteria = |c: &Ball<_>| c.cardinality() > 1;
        let root = Ball::par_new_tree(&data, &criteria, seed);

        let mut perm_data = data.clone();
        let perm_root = OffsetBall::par_from_ball_tree(root.clone(), &mut perm_data);

        utils::compare_permuted(
            c,
            metric_name,
            &data,
            &root,
            &perm_data,
            &perm_root,
            &queries,
            &radii,
            &ks,
        );
    }
}

criterion_group!(benches, genomic_search);
criterion_main!(benches);
