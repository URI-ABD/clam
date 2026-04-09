//! Tests for the `Cluster` struct.

#![expect(clippy::unwrap_used, clippy::cast_precision_loss)]

use abd_clam::{PartitionStrategy, Tree, common_metrics};
use test_case::test_case;

mod common;

#[test]
#[expect(clippy::expect_used)]
fn new() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
    let items = items.into_iter().enumerate().collect::<Vec<_>>(); // Convert to Vec<(usize, Vec<i32>)> to use index as metadata
    let cardinality = items.len();
    let metric = common_metrics::manhattan;

    // Don't partition in the root so we can run some tests.
    let tree = Tree::new(items.clone(), metric, &|_| (), &|_| false, &PartitionStrategy::default())?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(root.is_leaf(), "Root should be a leaf: {root:?}");
    assert_eq!(root.center_index(), 0, "Center index mismatch: {root:?}");
    assert_eq!(
        tree.iter_items().next().map(|(_, item, _)| item),
        Some(&vec![5, 6]),
        "Center mismatch: {root:?}"
    );
    assert_eq!(root.radius(), 12, "Radius mismatch: {root:?}");
    // Now partition the root
    let tree = Tree::new(items, metric, &|_| (), &|c| c.cardinality() > 2, &PartitionStrategy::default())?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");

    let subtree = tree.iter_clusters().collect::<Vec<_>>();
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
        r => unreachable!("Unexpected child radius: {r}, {:?}", subtree[1]),
    }

    // Check paths from root to items
    assert!(tree.path_to_item(0).is_some_and(|path| path.is_empty()), "Path to root center should be empty");
    let path_1 = tree.path_to_item(1).expect("Path to item 1 should exist");
    assert_eq!(path_1, vec![1]);

    let path_2 = tree.path_to_item(2).expect("Path to item 2 should exist");
    assert_eq!(path_2, vec![1, 2]);

    let path_3 = tree.path_to_item(3).expect("Path to item 3 should exist");
    assert_eq!(path_3, vec![3]);

    let path_4 = tree.path_to_item(4).expect("Path to item 4 should exist");
    assert_eq!(path_4, vec![3, 4]);

    Ok(())
}

#[test]
#[expect(clippy::expect_used)]
fn par_new() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
    let items = items.into_iter().enumerate().collect::<Vec<_>>(); // Convert to Vec<(usize, Vec<i32>)> to use index as metadata
    let cardinality = items.len();
    let metric = common_metrics::manhattan;

    // Don't partition in the root so we can run some tests.
    let tree = Tree::par_new(items.clone(), metric, &|_| (), &|_| false, &PartitionStrategy::default())?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(root.is_leaf(), "Root should be a leaf: {root:?}");
    assert_eq!(root.center_index(), 0, "Center index mismatch: {root:?}");
    assert_eq!(
        tree.iter_items().next().map(|(_, item, _)| item),
        Some(&vec![5, 6]),
        "Center mismatch: {root:?}"
    );
    assert_eq!(root.radius(), 12, "Radius mismatch: {root:?}");

    // Now partition the root
    let strategy = PartitionStrategy::binary();
    let tree = Tree::par_new(items, metric, &|_| (), &|c| c.cardinality() > 2, &strategy)?;
    let root = tree.root();

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");

    let subtree = tree.iter_clusters().collect::<Vec<_>>();
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
        r => unreachable!("Unexpected child radius: {r}, {:?}", subtree[1]),
    }

    // Check paths from root to items
    assert!(tree.path_to_item(0).is_some_and(|path| path.is_empty()), "Path to root center should be empty");
    let path_1 = tree.path_to_item(1).expect("Path to item 1 should exist");
    assert_eq!(path_1, vec![1]);

    let path_2 = tree.path_to_item(2).expect("Path to item 2 should exist");
    assert_eq!(path_2, vec![1, 2]);

    let path_3 = tree.path_to_item(3).expect("Path to item 3 should exist");
    assert_eq!(path_3, vec![3]);

    let path_4 = tree.path_to_item(4).expect("Path to item 4 should exist");
    assert_eq!(path_4, vec![3, 4]);

    Ok(())
}

#[test_case(10, 2 ; "10x2")]
#[test_case(100, 2 ; "100x2")]
#[test_case(100, 10 ; "100x10")]
fn big(car: usize, dim: usize) -> Result<(), String> {
    let metric = |a: &Vec<f32>, b: &Vec<f32>| {
        let d = common_metrics::euclidean(a, b);
        (d * 1000.0).trunc() / 1000.0 // Truncate to 3 decimal places to help debugging
    };
    let (min, max) = (-1.0, 1.0);
    let max_hypot = ((4 * dim) as f32).sqrt();
    let max_radius = 1.1 * max_hypot / 2.0; // Allow some slack for the radius due to the randomness of the data, and approximate geometric medians.

    for _ in 0..10 {
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

        let root = tree.root();
        assert_eq!(root.cardinality(), car, "Cardinality mismatch: {root:?}");
        assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
        assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");
        assert!(root.radius() <= max_radius, "Radius too large: {:.6} vs {max_radius}", root.radius());

        assert_eq!(tree.root().depth(), 0, "Root depth should be 0");
        for cluster in tree.iter_clusters() {
            if let Some(children) = tree.children_of(cluster) {
                for child in children {
                    assert_eq!(child.depth(), cluster.depth() + 1, "Child depth should be parent depth + 1",);
                }
            }
        }
    }

    Ok(())
}

#[test_case(1_000, 2 ; "1_000x2")]
#[test_case(1_000, 10 ; "1_000x10")]
fn par_big(car: usize, dim: usize) -> Result<(), String> {
    let metric = common_metrics::euclidean;
    let (min, max) = (-1.0, 1.0);
    let max_hypot = ((4 * dim) as f32).sqrt();
    let max_radius = 1.1 * max_hypot / 2.0; // Allow some slack for the radius due to the randomness of the data, and approximate geometric medians.

    for _ in 0..10 {
        let data = common::data_gen::tabular(car, dim, min, max);
        let tree = Tree::par_new_binary(data, metric)?;
        let n_clusters = tree.n_clusters();

        // These bounds were derived for large `car`
        let min_ratio = 2.0 / 3.0;
        let max_ratio = 1.0;
        let ratio = n_clusters as f64 / car as f64;

        assert!(
            (min_ratio..=max_ratio).contains(&ratio),
            "Unexpected number of clusters: {n_clusters} for {car} items (ratio: {ratio:.3}, expected range: [{min_ratio}, {max_ratio}])"
        );

        let root = tree.root();
        assert_eq!(root.cardinality(), car, "Cardinality mismatch: {root:?}");
        assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
        assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");
        assert!(root.radius() <= max_radius, "Radius too large: {:.6} vs {max_radius}", root.radius());

        assert_eq!(tree.root().depth(), 0, "Root depth should be 0");
        for cluster in tree.iter_clusters() {
            if let Some(children) = tree.children_of(cluster) {
                for child in children {
                    assert_eq!(child.depth(), cluster.depth() + 1, "Child depth should be parent depth + 1",);
                }
            }
        }
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
    for (n, v) in memo.iter_mut().enumerate().take(k + 1).skip(2) {
        *v = n - 1;
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

    let min_i = values
        .iter()
        .fold(None, |min_i, &((i, _), r)| {
            Some(match min_i {
                None => (i, r),
                Some((min_i, min_r)) => {
                    if (r < min_r) || ((r - min_r).abs() < f64::EPSILON && i < min_i) {
                        (i, r)
                    } else {
                        (min_i, min_r)
                    }
                }
            })
        })
        .map(|(i, _)| i)
        .unwrap();
    let ((min_i, min_v), min_r) = values[min_i - noisy_n];

    let max_i = values
        .iter()
        .fold(None, |max_i, &((i, _), r)| {
            Some(match max_i {
                None => (i, r),
                Some((max_i, max_r)) => {
                    if (r > max_r) || ((r - max_r).abs() < f64::EPSILON && i > max_i) {
                        (i, r)
                    } else {
                        (max_i, max_r)
                    }
                }
            })
        })
        .map(|(i, _)| i)
        .unwrap();
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
