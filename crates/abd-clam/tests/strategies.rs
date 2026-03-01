//! Tests for building trees with various partition strategies.

use abd_clam::{
    Cluster, NamedAlgorithm, PartitionStrategy, Tree,
    partition_strategy::{BranchingFactor, MaxSplit, SpanReductionFactor},
};
use test_case::test_case;

mod common;

type TestStrategy<T, A> = PartitionStrategy<fn(&Cluster<T, A>) -> bool>;
type MetricFnF32 = Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32 + Send + Sync>;

#[test_case(100, 2; "100x2")]
#[test_case(100, 10; "100x10")]
#[test_case(1_000, 2; "1000x2")]
#[test_case(1_000, 10; "1_000x10")]
fn strategy_suite(car: usize, dim: usize) -> Result<(), Box<dyn std::error::Error>> {
    let max_splits = [0.5, 0.95, 1.0];
    let branching_factors = [2, 4, 8, 16];
    let span_factors = [0.5, 0.75, 0.9];
    let strategies = make_suite(&max_splits, &branching_factors, &span_factors);

    let data = common::data_gen::tabular(car, dim, -1.0, 1.0);

    let metrics: Vec<MetricFnF32> = vec![
        Box::new(|a: &Vec<f32>, b: &Vec<f32>| distances::vectors::euclidean(a, b)),
        Box::new(|a: &Vec<f32>, b: &Vec<f32>| distances::vectors::manhattan(a, b)),
    ];

    for strategy in &strategies {
        for metric in &metrics {
            let items = data.iter().cloned().enumerate().collect();
            let tree = Tree::new(items, metric, strategy, &|_| ())?;
            check_tree(&tree, &strategy.name())?;

            let (items, _, metric) = tree.into_parts();
            let tree = Tree::par_new(items, metric, strategy, &|_| ())?;
            check_tree(&tree, &format!("par_{}", strategy.name()))?;
        }
    }

    Ok(())
}

fn make_suite<T, A>(max_splits: &[f64], branching_factors: &[usize], span_factors: &[f64]) -> Vec<TestStrategy<T, A>> {
    let mut strategies = Vec::new();

    for &ms in max_splits {
        assert!((0.5..=1.0).contains(&ms), "Max split factor must be between 0.5 and 1");
        strategies.push(PartitionStrategy::default().with_max_split(MaxSplit::Fixed(ms)));
    }
    strategies.push(PartitionStrategy::default().with_max_split(MaxSplit::NineTenths));
    strategies.push(PartitionStrategy::default().with_max_split(MaxSplit::ThreeQuarters));

    for &bf in branching_factors {
        assert!(bf > 1, "Branching factor must be greater than 1");
        strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Fixed(bf)));
        if bf > 4 {
            // Adaptive branching factor is only meaningful for larger values
            strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Adaptive(bf)));
        }
    }
    strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Logarithmic));

    for &sf in span_factors {
        assert!((f64::EPSILON..1.0).contains(&sf), "Span factor must be between epsilon and 1");
        strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Fixed(sf)));
    }
    strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Sqrt2));
    strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Two));
    strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::E));
    strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Pi));
    strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Phi));

    strategies
}

fn check_tree<Id, I, T, A, M>(tree: &Tree<Id, I, T, A, M>, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = tree.get_cluster(0)?;
    assert_eq!(root.center_index(), 0, "Root cluster ID should be 0 for strategy '{name}'");
    assert_eq!(
        root.cardinality(),
        tree.items().len(),
        "Root cluster should contain all items for strategy '{name}'"
    );

    assert!(
        tree.cluster_map().len() <= root.cardinality(),
        "Number of clusters should not exceed number of items for strategy '{name}'"
    );

    let mut visited = vec![false; root.cardinality()];
    for c in tree.sorted_clusters() {
        assert!(
            !visited[c.center_index()],
            "Cluster with ID {} is visited more than once for strategy '{name}'",
            c.center_index()
        );

        // Mark the center index as visited to ensure it is not reused in another cluster
        visited[c.center_index()] = true;

        if let Some(cids) = c.child_center_indices() {
            // This is a parent cluster

            // Check that the cluster has at least two children
            assert!(
                cids.len() > 1,
                "Cluster with ID {} has less than two children for strategy '{name}'",
                c.center_index()
            );

            // Check that the children have correct parent and depth, and accumulate their cardinality
            let mut total_cardinality = 0;
            for &cid in cids {
                let child = tree.get_cluster(cid)?;
                assert_eq!(
                    child.parent_center_index(),
                    Some(c.center_index()),
                    "Child cluster with ID {cid} has incorrect parent for strategy '{name}'"
                );
                assert_eq!(
                    child.depth(),
                    c.depth() + 1,
                    "Child cluster with ID {cid} has incorrect depth for strategy '{name}'"
                );
                total_cardinality += child.cardinality();
            }
            // Check that the total cardinality of the children matches the parent's cardinality (accounting for the center item)
            assert_eq!(
                total_cardinality + 1, // +1 for the center item
                c.cardinality(),
                "Cluster with ID {} has children with total cardinality {total_cardinality} that does not match its own cardinality {} for strategy '{name}'",
                c.center_index(),
                c.cardinality()
            );
        } else {
            // This is a leaf cluster

            if c.cardinality() > 1 {
                // All strategies start from the default, which creates leaf clusters of size at most 2, so if we see a leaf cluster with more than 1 item, it
                // must be because the strategy did not split it further, which means it should have exactly 2 items (the center and one other)
                assert_eq!(
                    c.cardinality(),
                    2,
                    "Leaf cluster with ID {} has cardinality greater than 2 for strategy '{name}'",
                    c.center_index()
                );
                // Mark the second item in the leaf cluster as visited
                visited[c.center_index() + 1] = true;
            }
        }
    }

    // Check that all items are covered by the clusters
    for (i, &v) in visited.iter().enumerate() {
        assert!(v, "Item with index {i} is not covered by any cluster for strategy '{name}'");
    }

    Ok(())
}
