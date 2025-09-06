//! Tests for the `Cluster` struct.

use abd_clam::{PartitionStrategy, Tree};
use ordered_float::OrderedFloat;
use test_case::test_case;

mod common;

#[test]
fn new() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
    let items = items.into_iter().enumerate().collect::<Vec<_>>();
    let cardinality = items.len();
    let metric = common::metrics::manhattan;

    // Don't partition in the root so we can run some tests.
    let strategy = PartitionStrategy::never();
    let tree = Tree::new(items.clone(), metric, &strategy, &|_| ())?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(root.is_leaf(), "Root should be a leaf: {root:?}");
    assert_eq!(tree.items()[root.center_index()].1, vec![5, 6], "Center mismatch: {root:?}");
    assert_eq!(root.radius(), 12, "Radius mismatch: {root:?}");
    // Now partition the root
    let strategy = PartitionStrategy::default().with_branching_factor(2.into());
    let tree = Tree::new(items, metric, &strategy, &|_| ())?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");

    let subtree: Vec<&abd_clam::Cluster<i32, ()>> = tree.sorted_clusters();
    if subtree.len() != 3 {
        eprintln!("{subtree:?}");
    }
    assert_eq!(
        subtree.len(),
        3, // Because both children will have cardinality == 2, and so will not be partitioned further
        "Subtree cardinality mismatch",
    );

    for child in &subtree[1..] {
        assert_eq!(child.cardinality(), 2, "Child cardinality mismatch: {child:?}");
        assert!(child.is_leaf(), "Child should be a leaf: {child:?}");
        assert_eq!(
            child.parent_center_index(),
            Some(root.center_index()),
            "Parent center index mismatch: {child:?}"
        );
        assert_ne!(
            child.center_index(),
            root.center_index(),
            "Child center index should not be equal to parent center index: {child:?}"
        );
    }
    match subtree[1].radius() {
        4 => assert_eq!(subtree[2].radius(), 8, "Child radius mismatch: {:?}", subtree[2]),
        8 => assert_eq!(subtree[2].radius(), 4, "Child radius mismatch: {:?}", subtree[2]),
        r => panic!("Unexpected child radius: {r}, {:?}", subtree[1]),
    }

    Ok(())
}

#[test]
fn par_new() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
    let items = items.into_iter().enumerate().collect::<Vec<_>>();
    let cardinality = items.len();
    let metric = common::metrics::manhattan;

    // Don't partition in the root so we can run some tests.
    let strategy = PartitionStrategy::never();
    let tree = Tree::par_new(items.clone(), metric, &strategy, &|_| ())?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(root.is_leaf(), "Root should be a leaf: {root:?}");
    assert_eq!(tree.items()[root.center_index()].1, vec![5, 6], "Center mismatch: {root:?}");
    assert_eq!(root.radius(), 12, "Radius mismatch: {root:?}");

    // Now partition the root
    let strategy = PartitionStrategy::default().with_branching_factor(2.into());
    let tree = Tree::par_new(items, metric, &strategy, &|_| ())?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");

    let subtree = tree.sorted_clusters();
    if subtree.len() != 3 {
        eprintln!("{subtree:?}");
    }
    assert_eq!(
        subtree.len(),
        3, // Because both children will have cardinality == 2, and so will not be partitioned further
        "Subtree cardinality mismatch",
    );

    for child in &subtree[1..] {
        assert_eq!(child.cardinality(), 2, "Child cardinality mismatch: {child:?}");
        assert!(child.is_leaf(), "Child should be a leaf: {child:?}");
        assert_eq!(
            child.parent_center_index(),
            Some(root.center_index()),
            "Parent center index mismatch: {child:?}"
        );
        assert_ne!(
            child.center_index(),
            root.center_index(),
            "Child center index should not be equal to parent center index: {child:?}"
        );
    }
    match subtree[1].radius() {
        4 => assert_eq!(subtree[2].radius(), 8, "Child radius mismatch: {:?}", subtree[2]),
        8 => assert_eq!(subtree[2].radius(), 4, "Child radius mismatch: {:?}", subtree[2]),
        r => panic!("Unexpected child radius: {r}, {:?}", subtree[1]),
    }

    Ok(())
}

#[test_case(10, 2 ; "10x2")]
#[test_case(1_000, 10 ; "1_000x10")]
fn big(car: usize, dim: usize) -> Result<(), String> {
    let metric = |a: &Vec<f32>, b: &Vec<f32>| {
        let d = common::metrics::euclidean::<_, _, f32>(a, b);
        (d * 1000.0).trunc() / 1000.0 // Truncate to 3 decimal places to help debugging
    };
    let (min, max) = (-1.0, 1.0);
    let max_hypot = ((4 * dim) as f32).sqrt();

    let mut ratios = Vec::new();
    for i in 0..10 {
        let data = common::data_gen::tabular(car, dim, min, max);
        let data = data
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .map(|f| (f * 1000.0).trunc() / 1000.0) // Truncate to 3 decimal places to help debugging
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let tree = Tree::new_minimal(data, metric)?;
        let n_clusters = tree.n_clusters();

        // These bounds were derived for large `car`
        let min_ratio = 2.0 / 3.0;
        let max_ratio = 1.0;
        let ratio = n_clusters as f64 / car as f64;

        assert!(
            (min_ratio..=max_ratio).contains(&ratio),
            "Unexpected number of clusters: {n_clusters} for {car} items (ratio: {ratio:.3}, expected range: [{min_ratio}, {max_ratio}])"
        );
        ratios.push(ratio);
        assert_eq!(ratios.len(), i + 1);

        let root = tree.root();
        assert_eq!(root.cardinality(), car, "Cardinality mismatch: {root:?}");
        assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
        assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");
        assert!(root.radius() <= max_hypot / 2.0, "Radius too large: {:.6}", root.radius());
    }

    Ok(())
}

#[test_case(1_000, 10 ; "1_000x10")]
#[test_case(10_000, 10 ; "10_000x10")]
fn par_big(car: usize, dim: usize) -> Result<(), String> {
    let metric = common::metrics::euclidean::<_, _, f32>;
    let (min, max) = (-1.0, 1.0);
    let hypot = ((4 * dim) as f32).sqrt();

    let mut ratios = Vec::new();
    for i in 0..10 {
        let data = common::data_gen::tabular(car, dim, min, max);
        let tree = Tree::par_new_minimal(data, metric)?;
        let n_clusters = tree.n_clusters();

        // These bounds were derived for large `car`
        let min_ratio = 2.0 / 3.0;
        let max_ratio = 1.0;
        let ratio = n_clusters as f64 / car as f64;

        assert!(
            (min_ratio..=max_ratio).contains(&ratio),
            "Unexpected number of clusters: {n_clusters} for {car} items (ratio: {ratio:.3}, expected range: [{min_ratio}, {max_ratio}])"
        );
        ratios.push(ratio);
        assert_eq!(ratios.len(), i + 1);

        let root = tree.root();
        assert_eq!(root.cardinality(), car, "Cardinality mismatch: {root:?}");
        assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
        assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");
        assert!(root.radius() <= hypot / 2.0, "Radius too large: {:.6}", root.radius());
    }

    Ok(())
}

#[test_case(2 ; "k=2")]
// #[test_case(4 ; "k=4")]
// #[test_case(8 ; "k=8")]
// #[test_case(10 ; "k=10")]
// #[test_case(16 ; "k=16")]
// #[test_case(32 ; "k=32")]
// #[test_case(64 ; "k=64")]
// #[test_case(100 ; "k=100")]
#[ignore = "This is just to gather some statistics about the number of clusters created in a binary tree"]
fn counting_clusters(k: usize) {
    let max_n = 1_000_000; // Max number of items
    let mut memo = vec![0; max_n + 1];

    // Recursive formula:
    // g(n) = 1 for n <= k
    // g(1 + i + kn) = 1 + i * g(n + 1) + (k - i) * g(n) for i in 0..k-1

    memo[0] = 1;
    memo[1] = 1;
    for n in 2..=k {
        memo[n] = n - 1;
    }
    for kn_i_1 in (k + 1)..=max_n {
        let r = (kn_i_1 - 1) % k;
        let q = (kn_i_1 - 1) / k;
        memo[kn_i_1] = 1 + r * memo[q + 1] + (k - r) * memo[q];
    }

    let noisy_n = 1023; // Skip small n where the ratio is very noisy
    let values = memo
        .into_iter()
        .enumerate()
        .skip(noisy_n)
        .map(|(i, v)| ((i, v), v as f64 / i as f64))
        .collect::<Vec<_>>();

    let min_i = values.iter().min_by_key(|&&((i, _), r)| (OrderedFloat(r), i)).map(|&((i, _), _)| i).unwrap();
    let ((min_i, min_v), min_r) = values[min_i - noisy_n];

    let max_i = values.iter().max_by_key(|&&((i, _), r)| (OrderedFloat(r), i)).map(|&((i, _), _)| i).unwrap();
    let ((max_i, max_v), max_r) = values[max_i - noisy_n];

    let mean_r = values.iter().map(|&(_, r)| r).sum::<f64>() / (values.len() as f64);
    let var_r = values.iter().map(|&(_, r)| (r - mean_r).powi(2)).sum::<f64>() / (values.len() as f64);
    let std_r = var_r.sqrt();

    let ((last_i, last_v), last_r) = values.last().copied().unwrap();

    let min_line = format!("  n with min ratio: {min_i}, num clusters: {min_v}, ratio: {min_r:.6}");
    let max_line = format!("  n with max ratio: {max_i}, num clusters: {max_v}, ratio: {max_r:.6}");
    let mean_line = format!("  mean ratio: {mean_r:.6}, std: {std_r:.6}");
    let last_line = format!("  final n = {last_i}, num clusters: {last_v}, ratio: {last_r:.6}");
    let output = format!("{min_line}\n{max_line}\n{mean_line}\n{last_line}");

    assert!(max_n == 0, "{output}"); // Always fails, just to print the output
}
