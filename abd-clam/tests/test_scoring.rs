use abd_clam::{Cakes, Cluster, Dataset, Graph, PartitionCriteria, Tree};
use std::collections::HashSet;
use std::fmt::Debug;

use abd_clam::chaoda::pretrained_models;

use abd_clam::cluster_selection::*;

mod utils;

#[test]
fn scoring() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);
    let metric = data.metric();

    let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
    let raw_tree = Tree::new(data, Some(42))
        .partition(&partition_criteria)
        .with_ratios(true);

    let mut root = raw_tree.root();

    let scorer_function = &pretrained_models::get_meta_ml_scorers()[0].1;

    let mut priority_queue = score_clusters(&root, scorer_function);

    assert_eq!(priority_queue.len(), root.subtree().len());

    let mut prev_value: f64;
    let mut curr_value: f64;

    prev_value = priority_queue.pop().unwrap().score;
    while !priority_queue.is_empty() {
        curr_value = priority_queue.pop().unwrap().score;
        assert!(prev_value >= curr_value);
        prev_value = curr_value;
    }

    let priority_queue = score_clusters(&root, scorer_function);
    let cluster_set = select_optimal_clusters(priority_queue).unwrap();
    for i in &cluster_set {
        for j in &cluster_set {
            if i != j {
                assert!(!i.is_descendant_of(j) && !i.is_ancestor_of(j));
            }
        }
    }

    for i in &root.subtree() {
        let mut ancestor_of = false;
        let mut descendant_of = false;
        for j in &cluster_set {
            if i.is_ancestor_of(j) {
                ancestor_of = true;
            }
            if i.is_descendant_of(j) {
                descendant_of = true
            }
        }
        assert!(ancestor_of || descendant_of || cluster_set.contains(i))
    }
}
