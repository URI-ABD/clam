//! Tests for the `Ball` struct.

use abd_clam::{
    cakes::PermutedBall,
    cluster::{
        adapter::{BallAdapter, ParBallAdapter},
        BalancedBall, ParPartition, Partition,
    },
    metric::{AbsoluteDifference, Manhattan},
    Ball, Cluster, Dataset, FlatVec, Metric,
};
use distances::{number::Multiplication, Number};

mod common;

#[test]
fn new() {
    let data = common::data_gen::gen_tiny_data();
    let metric = Manhattan;

    let indices = (0..data.cardinality()).collect::<Vec<_>>();
    let seed = Some(42);

    let root = Ball::new(&data, &metric, &indices, 0, seed).unwrap();
    common::cluster::test_new(&root, &data);

    let root = Ball::par_new(&data, &metric, &indices, 0, seed).unwrap();
    common::cluster::test_new(&root, &data);

    let root = BalancedBall::new(&data, &metric, &indices, 0, seed)
        .unwrap()
        .into_ball();
    common::cluster::test_new(&root, &data);

    let root = BalancedBall::par_new(&data, &metric, &indices, 0, seed)
        .unwrap()
        .into_ball();
    common::cluster::test_new(&root, &data);
}

#[test]
fn tree() {
    let data = common::data_gen::gen_tiny_data();
    let metric = Manhattan;

    let seed = Some(42);
    let criteria = |c: &Ball<_>| c.depth() < 1;

    let root = Ball::new_tree(&data, &metric, &criteria, seed);
    assert_eq!(root.indices().len(), data.cardinality());
    assert!(common::cluster::check_partition(&root));

    let root = Ball::par_new_tree(&data, &metric, &criteria, seed);
    assert_eq!(root.indices().len(), data.cardinality());
    assert!(common::cluster::check_partition(&root));

    let criteria = |c: &BalancedBall<_>| c.depth() < 1;
    let root = BalancedBall::new_tree(&data, &metric, &criteria, seed).into_ball();
    assert_eq!(root.indices().len(), data.cardinality());
    assert!(common::cluster::check_partition(&root));

    let root = BalancedBall::par_new_tree(&data, &metric, &criteria, seed).into_ball();
    assert_eq!(root.indices().len(), data.cardinality());
    assert!(common::cluster::check_partition(&root));
}

#[test]
fn partition_further() {
    let data = common::data_gen::gen_tiny_data();
    let metric = Manhattan;

    let seed = Some(42);
    let criteria_one = |c: &Ball<_>| c.depth() < 1;
    let criteria_two = |c: &Ball<_>| c.depth() < 2;

    let mut root = Ball::new_tree(&data, &metric, &criteria_one, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 1);
    }
    root.partition_further(&data, &metric, &criteria_two, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 2);
    }

    let mut root = Ball::par_new_tree(&data, &metric, &criteria_one, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 1);
    }
    root.par_partition_further(&data, &metric, &criteria_two, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 2);
    }

    let criteria_one = |c: &BalancedBall<_>| c.depth() < 1;
    let criteria_two = |c: &BalancedBall<_>| c.depth() < 2;

    let mut root = BalancedBall::new_tree(&data, &metric, &criteria_one, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 1);
    }
    root.partition_further(&data, &metric, &criteria_two, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 2);
    }

    let mut root = BalancedBall::par_new_tree(&data, &metric, &criteria_one, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 1);
    }
    root.par_partition_further(&data, &metric, &criteria_two, seed);
    for leaf in root.leaves() {
        assert_eq!(leaf.depth(), 2);
    }
}

#[test]
fn tree_iterative() {
    let data = common::data_gen::gen_pathological_line();
    let metric = AbsoluteDifference;

    let seed = Some(42);
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let indices = (0..data.cardinality()).collect::<Vec<_>>();
    let mut root = Ball::new(&data, &metric, &indices, 0, seed).unwrap();

    let depth_delta = abd_clam::utils::max_recursion_depth();
    let mut intermediate_depth = depth_delta;
    let intermediate_criteria = |c: &Ball<_>| c.depth() < intermediate_depth && criteria(c);
    root.partition(&data, &metric, &intermediate_criteria, seed);

    while root.leaves().into_iter().any(|l| !l.is_singleton()) {
        intermediate_depth += depth_delta;
        let intermediate_criteria = |c: &Ball<_>| c.depth() < intermediate_depth && criteria(c);
        root.partition_further(&data, &metric, &intermediate_criteria, seed);
    }

    assert!(!root.is_leaf());

    let criteria = |c: &BalancedBall<_>| c.cardinality() > 1;
    let mut root = BalancedBall::new(&data, &metric, &indices, 0, seed).unwrap();
    intermediate_depth = depth_delta;
    let intermediate_criteria = |c: &BalancedBall<_>| c.depth() < intermediate_depth && criteria(c);
    root.partition(&data, &metric, &intermediate_criteria, seed);

    while root.leaves().into_iter().any(|l| !l.is_singleton()) {
        intermediate_depth += depth_delta;
        let intermediate_criteria = |c: &BalancedBall<_>| c.depth() < intermediate_depth && criteria(c);
        root.partition_further(&data, &metric, &intermediate_criteria, seed);
    }

    assert!(!root.is_leaf());
}

#[test]
fn trim_and_graft() -> Result<(), String> {
    let line = (0..1024).collect();
    let metric = AbsoluteDifference;
    let data = FlatVec::new(line)?;

    let seed = Some(42);
    let criteria = |c: &Ball<_>| c.cardinality() > 1;
    let root = Ball::new_tree(&data, &metric, &criteria, seed);

    let target_depth = 4;
    let mut grafted_root = root.clone();
    let children = grafted_root.trim_at_depth(target_depth);

    let leaves = grafted_root.leaves();
    assert_eq!(leaves.len(), 2.powi(target_depth.as_i32()));
    assert_eq!(leaves.len(), children.len());

    grafted_root.graft_at_depth(target_depth, children);
    assert_eq!(grafted_root, root);
    for (l, c) in root.subtree().into_iter().zip(grafted_root.subtree()) {
        assert_eq!(l, c);
    }

    let criteria = |c: &BalancedBall<_>| c.cardinality() > 1;
    let root = BalancedBall::new_tree(&data, &metric, &criteria, seed);

    let target_depth = 4;
    let mut grafted_root = root.clone();
    let children = grafted_root.trim_at_depth(target_depth);

    let leaves = grafted_root.leaves();
    assert_eq!(leaves.len(), 2.powi(target_depth.as_i32()));
    assert_eq!(leaves.len(), children.len());

    grafted_root.graft_at_depth(target_depth, children);
    assert_eq!(grafted_root, root);
    for (l, c) in root.subtree().into_iter().zip(grafted_root.subtree()) {
        assert_eq!(l, c);
    }

    Ok(())
}

#[test]
fn permutation() {
    let data = common::data_gen::gen_tiny_data();
    let metric = Manhattan;

    let seed = Some(42);
    let criteria = |c: &Ball<_>| c.depth() < 1;

    let ball = Ball::new_tree(&data, &metric, &criteria, seed);

    let (root, perm_data) = PermutedBall::from_ball_tree(ball.clone(), data.clone(), &metric);
    assert!(check_permutation(&root, &perm_data, &metric));

    let (root, perm_data) = PermutedBall::par_from_ball_tree(ball, data, &metric);
    assert!(check_permutation(&root, &perm_data, &metric));
}

fn check_permutation<M: Metric<Vec<i32>, i32>>(
    root: &PermutedBall<i32, Ball<i32>>,
    data: &FlatVec<Vec<i32>, usize>,
    metric: &M,
) -> bool {
    assert!(!root.children().is_empty());

    for cluster in root.subtree() {
        let radius = data.one_to_one(cluster.arg_center(), cluster.arg_radial(), metric);
        assert_eq!(cluster.radius(), radius);
    }

    true
}
