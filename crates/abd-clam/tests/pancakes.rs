//! Tests for the `pancakes` module.

#![expect(dead_code, unused_variables, unused_imports)] // because some tests are disabled

use abd_clam::{
    cakes::{KnnBreadthFirst, KnnDepthFirst, KnnRepeatedRnn, RnnClustered},
    musals::{Aligner, CostMatrix},
    pancakes::{ParDecoder, ParEncoder, SquishedBall},
    Ball, Cluster, DiskIO, DistanceValue, ParDataset, ParPartition, Partition, Permutable,
};
use test_case::test_case;

mod common;

use common::metrics::levenshtein;

#[test_case(16, 16, 2)]
#[test_case(32, 16, 3)]
fn strings(num_clumps: usize, clump_size: usize, clump_radius: u16) -> Result<(), String> {
    let matrix = CostMatrix::<u16>::default_affine(Some(10));
    let aligner = Aligner::new(&matrix, b'-');

    let seed_length = 30;
    let alphabet = "ACTGN".chars().collect::<Vec<_>>();
    let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
    let penalties = distances::strings::Penalties::default();
    let inter_clump_distance_range = (clump_radius * 5, clump_radius * 7);
    let len_delta = seed_length / 10;
    let (_, sequences) = symagen::random_edits::generate_clumped_data(
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

    let query = seed_string.clone();
    let seed = Some(42);

    let radii = [1, 4, 8];
    let ks = [1, 10, 20];

    // TODO (Najib): Implement Encoder/Decoder for strings and re-enable pancakes search tests.
    // build_and_check_search(
    //     sequences,
    //     &levenshtein,
    //     &encoder,
    //     &decoder,
    //     &query,
    //     &radii,
    //     &ks,
    // );

    Ok(())
}

#[test]
fn ser_de() -> Result<(), String> {
    use abd_clam::Dataset;

    // The items.
    type I = i32;
    // The distance values.
    type T = i32;
    // The compressible dataset
    type Co = Vec<I>;
    // The ball for the compressible dataset.
    type B = Ball<T>;
    // The Encoder
    type Enc = ();
    // The Decoder
    type Dec = ();
    // The squishy ball
    type Sb = SquishedBall<I, T, Enc, Dec>;

    let mut data: Co = common::data_gen::gen_line_data(100);
    let metric = common::metrics::absolute_difference;

    let criteria = |c: &B| c.cardinality() > 1;
    let ball = B::new_tree(&data, &metric, &criteria);
    let (root, _) = Sb::from_cluster_tree(ball, &mut data, &metric, &(), 4);

    let co_data = root.decode_all(&())?;

    // let serialized = co_data.to_bytes()?;
    // let deserialized = Co::from_bytes(&serialized)?;

    // assert_eq!(co_data.cardinality(), deserialized.cardinality());

    Ok(())
}

/// Build trees and check the search results.
fn build_and_check_search<I, T, D, M, Enc, Dec>(
    mut data: D,
    metric: &M,
    encoder: &Enc,
    decoder: &Dec,
    query: &I,
    radii: &[T],
    ks: &[usize],
) -> Result<(), Dec::Err>
where
    I: core::fmt::Debug + Send + Sync + Eq + Clone,
    T: DistanceValue + core::fmt::Debug + Send + Sync,
    D: ParDataset<I> + Permutable<I> + Clone,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    Enc: ParEncoder<I, Dec>,
    Dec: ParDecoder<I, Enc>,
    Enc::Bytes: Send + Sync,
    Dec::Err: core::fmt::Debug + Send + Sync,
{
    let criterion = |c: &Ball<T>| c.cardinality() > 1;

    let (root, data) = {
        let ball = Ball::par_new_tree(&data, metric, &criterion);

        let (root, _) = SquishedBall::par_from_cluster_tree(ball, &mut data, metric, encoder, 4);
        let decoded_items = root.clone().par_decode_all(decoder)?;

        (root, decoded_items)
    };

    for &radius in radii {
        let alg = RnnClustered(radius);
        common::search::check_rnn(&root, &data, metric, query, radius, &alg, "RnnClustered");
    }

    for &k in ks {
        common::search::check_knn(
            &root,
            &data,
            metric,
            query,
            k,
            &KnnRepeatedRnn(k, T::one() + T::one()),
            "KnnRepeatedRnn",
        );
        common::search::check_knn(&root, &data, metric, query, k, &KnnBreadthFirst(k), "KnnBreadthFirst");
        common::search::check_knn(&root, &data, metric, query, k, &KnnDepthFirst(k), "KnnDepthFirst");
    }

    Ok(())
}
