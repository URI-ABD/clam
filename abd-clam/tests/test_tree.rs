//! Tests on the tree module.

use abd_clam::{Cluster, Dataset, Instance, PartitionCriteria, Tree, VecDataset};
use distances::Number;
use tempdir::TempDir;

mod utils;

#[test]
fn leaf_indices() {
    let data = utils::gen_dataset_from(
        vec![
            vec![10.],
            vec![1.],
            vec![-5.],
            vec![8.],
            vec![3.],
            vec![2.],
            vec![0.5],
            vec![0.],
        ],
        utils::euclidean::<f32, f32>,
        None,
    );
    let partition_criteria = PartitionCriteria::default();

    let tree = Tree::new(data, Some(42)).partition(&partition_criteria);

    let leaf_indices = tree.root().indices().collect::<Vec<_>>();
    let tree_indices = (0..tree.cardinality()).collect::<Vec<_>>();

    assert_eq!(leaf_indices, tree_indices);
}

#[test]
fn reordering() {
    let data = utils::gen_dataset_from(
        vec![
            vec![10.],
            vec![1.],
            vec![-5.],
            vec![8.],
            vec![3.],
            vec![2.],
            vec![0.5],
            vec![0.],
        ],
        utils::euclidean::<f32, f32>,
        None,
    );
    let partition_criteria = PartitionCriteria::default();

    let tree = Tree::new(data, Some(42)).partition(&partition_criteria);

    let tree_indices = (0..tree.cardinality()).collect::<Vec<_>>();
    // Assert that the root's indices actually cover the whole dataset.
    assert_eq!(tree.data().cardinality(), tree_indices.len());

    // Assert that the tree's indices have been reordered in depth-first order
    assert_eq!((0..tree.cardinality()).collect::<Vec<_>>(), tree_indices);
}

#[test]
fn save_load() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);
    let metric = data.metric();

    let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
    let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria);

    let tree_dir = TempDir::new("tree_medium").unwrap();

    // Save the tree
    raw_tree.save(tree_dir.path()).unwrap();

    // Recover the tree
    let rec_tree = Tree::load(tree_dir.path(), metric, false).unwrap();

    // Assert recovering was successful
    assert_eq!(raw_tree.depth(), rec_tree.depth(), "Tree depths not equal.");
    assert_subtree_equal(
        raw_tree.root(),
        raw_tree.data(),
        rec_tree.root(),
        rec_tree.data(),
        metric,
    );
}

/// Asserts that two clusters are equal.
fn assert_subtree_equal<I: Instance, U: Number>(
    raw_cluster: &Cluster<U>,
    raw_data: &VecDataset<I, U, bool>,
    rec_cluster: &Cluster<U>,
    rec_data: &VecDataset<I, U, bool>,
    metric: fn(&I, &I) -> U,
) {
    // Assert their cardinalities
    assert_eq!(
        raw_cluster.cardinality(),
        rec_cluster.cardinality(),
        "Cardinalities are not equal."
    );

    // Resolve centers
    let (raw_center, rec_center) = (&raw_data[raw_cluster.arg_center()], &rec_data[rec_cluster.arg_center()]);
    let (raw_radial, rec_radial) = (&raw_data[raw_cluster.arg_radial()], &rec_data[rec_cluster.arg_radial()]);

    // Assert centers and radials are equal
    assert_eq!(metric(raw_center, rec_center), U::zero(), "Centers are not equal.");
    assert_eq!(metric(raw_radial, rec_radial), U::zero(), "Radials are not equal.");

    // Get children and assert they are of equal optionality
    let (raw_children, rec_children) = (&raw_cluster.children(), &rec_cluster.children());

    match raw_children {
        None => assert!(rec_children.is_none(), "One cluster has children, the other does not"),
        Some([left_1, right_1]) => {
            assert!(rec_children.is_some(), "One cluster has children, the other does not");

            let [left_2, right_2] = rec_children.unwrap();

            assert_subtree_equal(left_1, raw_data, left_2, rec_data, metric);
            assert_subtree_equal(right_1, raw_data, right_2, rec_data, metric);
        }
    }
}

#[test]
fn get_cluster() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);

    let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
    let tree = Tree::new(data, Some(42)).partition(&partition_criteria);

    let clusters = tree.root().subtree();

    for d in 0..tree.depth() {
        for &c in clusters.iter().filter(|c| c.depth() == d) {
            let (offset, cardinality) = (c.offset(), c.cardinality());

            let c_ = tree.get_cluster(offset, cardinality);
            assert!(c_.is_some());

            let c_ = c_.unwrap();
            assert_eq!(c_, c);
        }
    }
}
