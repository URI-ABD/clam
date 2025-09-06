//! Tests for the `Ball` struct.

use abd_clam::{cakes::PermutedBall, Ball, Cluster, Dataset, ParPartition, Partition};

mod common;

#[test]
fn new() {
    let data = common::data_gen::gen_tiny_data();
    let metric = common::metrics::manhattan;

    let indices = (0..data.cardinality()).collect::<Vec<_>>();

    let root = Ball::new(&data, &metric, &indices, 0).unwrap();
    common::cluster::test_new(&root, &data);

    let root = Ball::par_new(&data, &metric, &indices, 0).unwrap();
    common::cluster::test_new(&root, &data);

    // let root = BalancedBall::new(&data, &metric, &indices, 0).unwrap().into_ball();
    // common::cluster::test_new(&root, &data);

    // let root = BalancedBall::par_new(&data, &metric, &indices, 0).unwrap().into_ball();
    // common::cluster::test_new(&root, &data);
}

#[test]
fn tree() {
    let data = common::data_gen::gen_tiny_data();
    let metric = common::metrics::manhattan;
    let criteria = |c: &Ball<_>| c.depth() < 1;

    let root = Ball::new_tree(&data, &metric, &criteria);
    assert_eq!(root.indices().len(), data.cardinality(), "{root:?}");
    assert!(common::cluster::check_partition(&root));

    let root = Ball::par_new_tree(&data, &metric, &criteria);
    assert_eq!(root.indices().len(), data.cardinality(), "{root:?}");
    assert!(common::cluster::check_partition(&root));

    // let criteria = |c: &BalancedBall<_>| c.depth() < 1;
    // let root = BalancedBall::new_tree(&data, &metric, &criteria).into_ball();
    // assert_eq!(root.indices().len(), data.cardinality());
    // assert!(common::cluster::check_partition(&root));

    // let root = BalancedBall::par_new_tree(&data, &metric, &criteria).into_ball();
    // assert_eq!(root.indices().len(), data.cardinality());
    // assert!(common::cluster::check_partition(&root));
}

#[test]
fn tree_iterative() {
    let data = common::data_gen::gen_pathological_line();
    let metric = common::metrics::absolute_difference;
    let depth_stride = abd_clam::utils::max_recursion_depth();

    let root = Ball::new_tree_iterative(&data, &metric, &|_| true, depth_stride);
    assert!(!root.is_leaf());

    // let root = BalancedBall::new_tree_iterative(&data, &metric, &|_| true, depth_stride);
    // assert!(!root.is_leaf());
}

#[test]
fn trim_and_graft() -> Result<(), String> {
    let metric = common::metrics::absolute_difference;
    let data = (0..1024).collect::<Vec<_>>();
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let root = Ball::new_tree(&data, &metric, &criteria);

    let target_depth = 4;
    let mut grafted_root = root.clone();
    let children = grafted_root.trim_at_depth(target_depth);

    let leaves = grafted_root.leaves();
    assert_eq!(leaves.len(), 2_usize.pow(target_depth as u32));
    assert_eq!(leaves.len(), children.len());

    grafted_root.graft_at_depth(target_depth, children);
    assert_eq!(grafted_root, root);
    for (l, c) in root.subtree().into_iter().zip(grafted_root.subtree()) {
        assert_eq!(l, c);
    }

    // let criteria = |c: &BalancedBall<_>| c.cardinality() > 1;
    // let root = BalancedBall::new_tree(&data, &metric, &criteria);

    let target_depth = 4;
    let mut grafted_root = root.clone();
    let children = grafted_root.trim_at_depth(target_depth);

    let leaves = grafted_root.leaves();
    assert_eq!(leaves.len(), 2_usize.pow(target_depth as u32));
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
    let mut data = common::data_gen::gen_tiny_data();
    let metric = common::metrics::manhattan;
    let criteria = |c: &Ball<_>| c.depth() < 1;

    let ball = Ball::new_tree(&data, &metric, &criteria);

    let mut perm_data = data.clone();
    let (root, _) = PermutedBall::from_cluster_tree(ball.clone(), &mut perm_data);
    assert!(check_permutation(&root, &perm_data, &metric));

    let (root, _) = PermutedBall::par_from_cluster_tree(ball, &mut data);
    assert!(check_permutation(&root, &data, &metric));
}

fn check_permutation<M: Fn(&Vec<i32>, &Vec<i32>) -> i32>(
    root: &PermutedBall<i32, Ball<i32>>,
    data: &Vec<Vec<i32>>,
    metric: &M,
) -> bool {
    assert!(!root.children().is_empty());

    for cluster in root.subtree() {
        let radius = data.one_to_one(cluster.arg_center(), cluster.arg_radial(), metric);
        assert_eq!(cluster.radius(), radius, "Cluster: {cluster:?}");
    }

    true
}
