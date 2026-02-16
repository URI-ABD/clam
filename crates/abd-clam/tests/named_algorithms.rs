//! Ensure that named algorithms can be converted to/from their string representations.

use abd_clam::{
    NamedAlgorithm, PartitionStrategy,
    cakes::{
        self, Cakes, MeasurableSearchQuality,
        quality::{Recall, RelativeDistanceError},
    },
    musals::{
        MeasurableAlignmentQuality,
        quality::{DistanceDistortion, FragmentationRate, GapFraction, SumOfPairs},
    },
    tree::partition_strategy::{BranchingFactor, MaxFraction, SpanReductionFactor},
};

/// Check that the given algorithm can be converted to a string and back without loss of information.
fn check_round_trip<Alg>(alg: &Alg, expected_repr: &str)
where
    Alg: NamedAlgorithm,
{
    let repr = alg.to_string();
    assert_eq!(repr, expected_repr, "Algorithm string representation does not match expected value");

    #[expect(clippy::panic, reason = "This is a test.")]
    let parsed = Alg::from_str(&repr).unwrap_or_else(|_| panic!("Failed to parse algorithm: Original: {alg:?}, String: {repr}, Expected: {expected_repr}"));
    assert_eq!(format!("{parsed:?}"), format!("{alg:?}"), "Parsed algorithm does not match original");

    let new_repr = parsed.to_string();
    assert_eq!(new_repr, expected_repr, "Round-trip conversion did not yield the same string representation");
}

/// Check that all parsing is successful for all distance types for the given algorithm.
macro_rules! check_knn_for_types {
    ($variant:ident, $alg:expr, $repr:expr) => {
        check_round_trip(&$alg, &$repr);

        check_round_trip(&Cakes::<f32>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<f64>::$variant($alg), &$repr);

        check_round_trip(&Cakes::<i8>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<i16>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<i32>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<i64>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<isize>::$variant($alg), &$repr);

        check_round_trip(&Cakes::<u8>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<u16>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<u32>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<u64>::$variant($alg), &$repr);
        check_round_trip(&Cakes::<usize>::$variant($alg), &$repr);
    };
}

#[test]
fn cakes_names() {
    // Check ranged search algorithms with floating-point parameters.
    for r in [0.1, 0.5, 0.9, 1_f32, 1.5, 2.0, 10.0, 20.0, 100.0] {
        let repr = format!("rnn-chess::radius={r}");
        check_round_trip(&cakes::RnnChess::new(r), &repr);
        check_round_trip(&Cakes::RnnChess(cakes::RnnChess::new(r)), &repr);

        let repr = format!("rnn-linear::radius={r}");
        check_round_trip(&cakes::RnnLinear::new(r), &repr);
        check_round_trip(&Cakes::RnnLinear(cakes::RnnLinear::new(r)), &repr);
    }
    for r in [0.1, 0.5, 0.9, 1_f64, 1.5, 2.0, 10.0, 20.0, 100.0] {
        let repr = format!("rnn-chess::radius={r}");
        check_round_trip(&cakes::RnnChess::new(r), &repr);
        check_round_trip(&Cakes::RnnChess(cakes::RnnChess::new(r)), &repr);

        let repr = format!("rnn-linear::radius={r}");
        check_round_trip(&cakes::RnnLinear::new(r), &repr);
        check_round_trip(&Cakes::RnnLinear(cakes::RnnLinear::new(r)), &repr);
    }

    // Check all search algorithms with integer parameters.
    for k in [1, 5, 10, 20, 100] {
        check_knn_for_types!(KnnBfs, cakes::KnnBfs::new(k), format!("knn-bfs::k={k}"));
        check_knn_for_types!(KnnDfs, cakes::KnnDfs::new(k), format!("knn-dfs::k={k}"));
        check_knn_for_types!(KnnLinear, cakes::KnnLinear::new(k), format!("knn-linear::k={k}"));
        check_knn_for_types!(KnnRrnn, cakes::KnnRrnn::new(k), format!("knn-rrnn::k={k}"));
        check_knn_for_types!(KnnSieve, cakes::KnnSieve::new(k), format!("knn-sieve::k={k}"));

        for tol in [0.01, 0.1, 0.5] {
            check_knn_for_types!(
                ApproxKnnBfs,
                cakes::approximate::KnnBfs::new(k, tol),
                format!("approx-knn-bfs::k={k},tol={tol}")
            );
            check_knn_for_types!(
                ApproxKnnDfs,
                cakes::approximate::KnnDfs::new(k, tol),
                format!("approx-knn-dfs::k={k},tol={tol}")
            );
            check_knn_for_types!(
                ApproxKnnSieve,
                cakes::approximate::KnnSieve::new(k, tol),
                format!("approx-knn-sieve::k={k},tol={tol}")
            );
        }
    }
}

#[test]
fn cakes_quality_names() {
    let recall = Recall;
    let repr = "recall";
    check_round_trip(&recall, repr);
    check_round_trip(&MeasurableSearchQuality::Recall, repr);
    check_round_trip(&MeasurableSearchQuality::from(recall), repr);

    let rde = RelativeDistanceError;
    let repr = "relative-distance-error";
    check_round_trip(&rde, repr);
    check_round_trip(&MeasurableSearchQuality::RelativeDistanceError, repr);
    check_round_trip(&MeasurableSearchQuality::from(rde), repr);
}

#[test]
fn musals_quality_names() {
    let dd = DistanceDistortion;
    let repr = "distance-distortion".to_string();
    check_round_trip(&dd, &repr);
    check_round_trip(&MeasurableAlignmentQuality::DistanceDistortion, &repr);
    check_round_trip(&MeasurableAlignmentQuality::from(dd), &repr);

    let fr = FragmentationRate;
    let repr = "fragmentation-rate".to_string();
    check_round_trip(&fr, &repr);
    check_round_trip(&MeasurableAlignmentQuality::FragmentationRate, &repr);
    check_round_trip(&MeasurableAlignmentQuality::from(fr), &repr);

    let gf = GapFraction;
    let repr = "gap-fraction".to_string();
    check_round_trip(&gf, &repr);
    check_round_trip(&MeasurableAlignmentQuality::GapFraction, &repr);
    check_round_trip(&MeasurableAlignmentQuality::from(gf), &repr);

    let sop = SumOfPairs;
    let repr = "sum-of-pairs".to_string();
    check_round_trip(&sop, &repr);
    check_round_trip(&MeasurableAlignmentQuality::SumOfPairs, &repr);
    check_round_trip(&MeasurableAlignmentQuality::from(sop), &repr);
}

#[test]
fn strategy_names() {
    let mut named_branching_factors = vec![
        (BranchingFactor::Binary, "branching-factor::binary".to_string()),
        (BranchingFactor::Logarithmic, "branching-factor::logarithmic".to_string()),
    ];
    for bf in [2, 4, 8, 16] {
        named_branching_factors.push((BranchingFactor::Fixed(bf), format!("branching-factor::fixed={bf}")));
        named_branching_factors.push((BranchingFactor::Adaptive(bf), format!("branching-factor::adaptive={bf}")));
    }

    for (bf, repr) in named_branching_factors {
        check_round_trip(&bf, &repr);
        check_round_trip(&PartitionStrategy::BranchingFactor(bf), &repr);
        check_round_trip(&PartitionStrategy::from(bf), &repr);
    }

    let mut named_max_fractions = vec![
        (MaxFraction::NineTenths, "max-fraction::nine-tenths".to_string()),
        (MaxFraction::ThreeQuarters, "max-fraction::three-quarters".to_string()),
    ];
    for mf in [0.1, 0.5, 0.9] {
        named_max_fractions.push((MaxFraction::Fixed(mf), format!("max-fraction::fixed={mf}")));
    }

    for (mf, repr) in named_max_fractions {
        check_round_trip(&mf, &repr);
        check_round_trip(&PartitionStrategy::MaxFraction(mf), &repr);
        check_round_trip(&PartitionStrategy::from(mf), &repr);
    }

    let mut named_span_reduction_factors = vec![
        (SpanReductionFactor::Sqrt2, "span-reduction-factor::sqrt2".to_string()),
        (SpanReductionFactor::Two, "span-reduction-factor::two".to_string()),
        (SpanReductionFactor::E, "span-reduction-factor::e".to_string()),
        (SpanReductionFactor::Pi, "span-reduction-factor::pi".to_string()),
        (SpanReductionFactor::GoldenRatio, "span-reduction-factor::golden-ratio".to_string()),
    ];
    for srf in [1.5, 5.0, 10.0] {
        named_span_reduction_factors.push((SpanReductionFactor::Fixed(srf), format!("span-reduction-factor::fixed={srf}")));
    }

    for (srf, repr) in named_span_reduction_factors {
        check_round_trip(&srf, &repr);
        check_round_trip(&PartitionStrategy::SpanReductionFactor(srf), &repr);
        check_round_trip(&PartitionStrategy::from(srf), &repr);
    }
}
