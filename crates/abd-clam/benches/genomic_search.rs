//! Benchmark for genomic search.

use std::collections::HashMap;

use abd_clam::{
    cakes::{HintedDataset, PermutedBall},
    cluster::{adapter::ParBallAdapter, BalancedBall, ParPartition},
    dataset::{AssociatesMetadata, AssociatesMetadataMut},
    metric::Levenshtein,
    msa::{Aligner, CostMatrix, Sequence},
    pancakes::SquishyBall,
    Ball, Cluster, Dataset, FlatVec,
};
use criterion::*;
use rand::prelude::*;

mod utils;

fn genomic_search(c: &mut Criterion) {
    let matrix = CostMatrix::<u16>::default();
    let aligner = Aligner::new(&matrix, b'-');

    let seed_length = 250;
    let alphabet = "ACTGN".chars().collect::<Vec<_>>();
    let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
    let penalties = distances::strings::Penalties::default();
    let num_clumps = 500;
    let clump_size = 20;
    let clump_radius = 10_u16;
    let inter_clump_distance_range = (50_u16, 80_u16);
    let len_delta = 10;
    let (metadata, genomes) = symagen::random_edits::generate_clumped_data(
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
    .map(|(m, seq)| (m, Sequence::new(seq, Some(&aligner))))
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
    let radii = vec![1_u32, 5, 10];
    let ks = vec![1, 10, 100];

    let data = FlatVec::new(genomes)
        .unwrap()
        .with_metadata(&metadata)
        .unwrap()
        .with_name("genomic-search");
    let metric = Levenshtein;

    let criteria = |c: &Ball<_>| c.cardinality() > 1;
    let ball = Ball::par_new_tree(&data, &metric, &criteria, seed);

    let criteria = |c: &BalancedBall<_>| c.cardinality() > 1;
    let balanced_ball = BalancedBall::par_new_tree(&data, &metric, &criteria, seed).into_ball();

    let (_, max_radius) = abd_clam::utils::arg_max(&radii).unwrap();
    let (_, max_k) = abd_clam::utils::arg_max(&ks).unwrap();
    let data = data
        .transform_metadata(|s| (s.clone(), HashMap::new()))
        .with_hints_from_tree(&ball, &metric)
        .with_hints_from(&metric, &balanced_ball, max_radius, max_k);

    let (perm_ball, perm_data) = PermutedBall::par_from_ball_tree(ball.clone(), data.clone(), &metric);
    let (squishy_ball, dec_data) = SquishyBall::par_from_ball_tree(ball.clone(), data.clone(), &metric);
    let dec_data = dec_data.with_metadata(perm_data.metadata()).unwrap();

    let (perm_balanced_ball, perm_balanced_data) =
        PermutedBall::par_from_ball_tree(balanced_ball.clone(), data.clone(), &metric);
    let (squishy_balanced_ball, dec_balanced_data) =
        SquishyBall::par_from_ball_tree(balanced_ball.clone(), data.clone(), &metric);
    let dec_balanced_data = dec_balanced_data.with_metadata(perm_balanced_data.metadata()).unwrap();

    utils::compare_permuted(
        c,
        &metric,
        (&ball, &data),
        (&balanced_ball, &perm_balanced_data),
        (&perm_ball, &perm_data),
        (&perm_balanced_ball, &perm_balanced_data),
        Some((&squishy_ball, &dec_data)),
        Some((&squishy_balanced_ball, &dec_balanced_data)),
        &queries,
        &radii,
        &ks,
        true,
    );
}

criterion_group!(benches, genomic_search);
criterion_main!(benches);
