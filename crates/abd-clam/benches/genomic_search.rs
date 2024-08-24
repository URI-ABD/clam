//! Benchmark for genomic search.

mod utils;

use abd_clam::{
    adapter::{Adapter, ParAdapter, ParBallAdapter},
    cakes::{OffBall, SquishyBall},
    partition::ParPartition,
    BalancedBall, Ball, Cluster, FlatVec, Metric, Permutable,
};
use criterion::*;
use rand::prelude::*;

pub use utils::read_ann_data_npy;

const METRICS: &[(&str, fn(&String, &String) -> u64)] = &[
    ("levenshtein", |x: &String, y: &String| {
        distances::strings::levenshtein(x, y)
    }),
    // ("needleman-wunsch", |x: &String, y: &String| {
    //     distances::strings::nw_distance(x, y)
    // }),
];

fn genomic_search(c: &mut Criterion) {
    let seed_length = 250;
    let alphabet = "ACTGN".chars().collect::<Vec<_>>();
    let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
    let penalties = distances::strings::Penalties::default();
    let num_clumps = 500;
    let clump_size = 20;
    let clump_radius = 10_u16;
    let inter_clump_distance_range = (50_u16, 80_u16);
    let len_delta = 10;
    let (_, genomes) = symagen::random_edits::generate_clumped_data(
        &seed_string,
        penalties,
        &alphabet,
        num_clumps,
        clump_size,
        clump_radius,
        inter_clump_distance_range,
        len_delta,
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
    let radii = vec![];
    let ks = vec![1, 10, 20];
    for &(metric_name, distance_fn) in METRICS {
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(genomes.clone(), metric).unwrap();

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let ball = Ball::par_new_tree(&data, &criteria, seed);
        let (off_ball, perm_data) = OffBall::par_from_ball_tree(ball.clone(), data.clone());
        let (squishy_ball, dec_data) = SquishyBall::par_from_ball_tree(ball.clone(), data.clone());

        let criteria = |c: &BalancedBall<_, _, _>| c.cardinality() > 1;
        let balanced_ball = BalancedBall::par_new_tree(&data, &criteria, seed);
        let (balanced_off_ball, balanced_perm_data) = {
            let balanced_off_ball = OffBall::par_adapt_tree(balanced_ball.clone(), None);
            let mut balanced_perm_data = data.clone();
            let permutation = balanced_off_ball.source().indices().collect::<Vec<_>>();
            balanced_perm_data.permute(&permutation);
            (balanced_off_ball, balanced_perm_data)
        };

        utils::compare_permuted(
            c,
            "genomic-search",
            metric_name,
            (&ball, &data),
            (&off_ball, &perm_data),
            Some((&squishy_ball, &dec_data)),
            (&balanced_ball, &data),
            (&balanced_off_ball, &balanced_perm_data),
            None,
            &queries,
            &radii,
            &ks,
            true,
        );
    }
}

criterion_group!(benches, genomic_search);
criterion_main!(benches);
