//! Tests for the `pancakes` module.

use distances::Number;
use test_case::test_case;

use abd_clam::{
    cakes::{KnnBreadthFirst, KnnDepthFirst, KnnRepeatedRnn, PermutedBall, RnnClustered},
    cluster::{
        adapter::{Adapter, BallAdapter, ParAdapter},
        ParPartition, Partition,
    },
    dataset::{AssociatesMetadata, AssociatesMetadataMut, Permutable},
    metric::{AbsoluteDifference, Levenshtein, ParMetric},
    msa::{Aligner, CostMatrix, Sequence},
    pancakes::{CodecData, Decodable, Encodable, ParCompressible, SquishyBall},
    Ball, Cluster, FlatVec,
};

mod common;

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
    let (metadata, data) = symagen::random_edits::generate_clumped_data(
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

    let data = FlatVec::new(data)?.with_metadata(&metadata)?;
    let query = Sequence::new(seed_string.clone(), Some(&aligner));
    let seed = Some(42);

    let radii = [1, 4, 8];
    let ks = [1, 10, 20];

    build_and_check_search(&data, &Levenshtein, &query, seed, &radii, &ks);

    Ok(())
}

#[cfg(feature = "disk-io")]
#[test]
fn ser_de() -> Result<(), String> {
    use abd_clam::Dataset;

    // The items.
    type I = i32;
    // The distance values.
    type U = i32;
    // The compressible dataset
    type Co = FlatVec<I, usize>;
    // The ball for the compressible dataset.
    type B = Ball<U>;
    // The decompressible dataset
    type Dec = CodecData<I, usize>;
    // The squishy ball
    type Sb = SquishyBall<U, B>;

    let data: Co = common::data_gen::gen_line_data(100);
    let metric = AbsoluteDifference;
    let metadata = data.metadata().to_vec();

    let criteria = |c: &B| c.cardinality() > 1;
    let ball = B::new_tree(&data, &metric, &criteria, Some(42));
    let (_, co_data) = Sb::from_ball_tree(ball, data, &metric);
    let co_data = co_data.with_metadata(&metadata)?;

    let serialized = bitcode::encode(&co_data).map_err(|e| e.to_string())?;
    let deserialized: Dec = bitcode::decode(&serialized).map_err(|e| e.to_string())?;

    assert_eq!(co_data.cardinality(), deserialized.cardinality());
    assert_eq!(co_data.dimensionality_hint(), deserialized.dimensionality_hint());
    assert_eq!(co_data.metadata(), deserialized.metadata());
    assert_eq!(co_data.permutation(), deserialized.permutation());
    assert_eq!(co_data.center_map(), deserialized.center_map());
    assert_eq!(co_data.leaf_bytes(), deserialized.leaf_bytes());

    Ok(())
}

/// Build trees and check the search results.
fn build_and_check_search<I, T, D, M>(data: &D, metric: &M, query: &I, seed: Option<u64>, radii: &[T], ks: &[usize])
where
    I: core::fmt::Debug + Send + Sync + Encodable + Decodable + Clone + Eq,
    T: Number,
    D: ParCompressible<I> + Permutable + Clone,
    M: ParMetric<I, T>,
{
    let criterion = |c: &Ball<T>| c.cardinality() > 1;

    let (ball, data) = {
        let ball = Ball::par_new_tree(data, metric, &criterion, seed);

        let mut perm_data = data.clone();
        let mut perm_ball = PermutedBall::par_adapt_tree_iterative(ball, None, &perm_data, metric);

        let permutation = <PermutedBall<T, _> as Adapter<I, T, D, D, _, _>>::source(&perm_ball).indices();
        perm_data.permute(&permutation);
        perm_ball.clear_source_indices();

        let mut squishy_ball =
            <SquishyBall<_, _> as ParAdapter<_, _, _, CodecData<I, usize>, _, _>>::par_adapt_tree_iterative(
                perm_ball, None, &perm_data, metric,
            );

        squishy_ball.par_set_costs(&perm_data, metric);
        squishy_ball.trim(4);

        let co_data = CodecData::par_from_compressible(&perm_data, &squishy_ball);

        (squishy_ball, co_data)
    };

    for &radius in radii {
        let alg = RnnClustered(radius);
        common::search::check_rnn(&ball, &data, metric, query, radius, &alg);
    }

    for &k in ks {
        common::search::check_knn(&ball, &data, metric, query, k, &KnnRepeatedRnn(k, T::ONE.double()));
        common::search::check_knn(&ball, &data, metric, query, k, &KnnBreadthFirst(k));
        common::search::check_knn(&ball, &data, metric, query, k, &KnnDepthFirst(k));
    }
}
