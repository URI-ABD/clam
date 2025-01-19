//! Tests for the `Tree` struct.

use abd_clam::{cluster::Partition, metric::Euclidean, Ball, Cluster, Tree};
use test_case::test_case;

mod common;

#[test_case(20, 2)]
#[test_case(1_000, 10)]
#[test_case(10_000, 10)]
fn from_root(car: usize, dim: usize) {
    let max = 1.0;
    let seed = 42;

    let data = common::data_gen::gen_random_data(car, dim, max, seed);
    let metric = Euclidean;
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let root = Ball::new_tree(&data, &metric, &criteria, Some(seed));
    let true_subtree = {
        let mut subtree = root.subtree();
        subtree.sort();
        subtree
    };

    // let tree = Tree::new(&data, &metric, &criteria, Some(seed));
    let tree = Tree::from(root.clone());
    let tree_subtree = {
        let mut subtree = tree.bft().collect::<Vec<_>>();
        subtree.sort();
        subtree
    };

    assert_eq!(true_subtree.len(), tree_subtree.len());
    for (&a, &b) in true_subtree.iter().zip(tree_subtree.iter()) {
        check_ball_eq(a, b);
    }

    for a in true_subtree {
        let res = tree.find(a);
        assert!(res.is_some());
        let (depth, index, _, _) = res.unwrap();

        let b = tree.get(depth, index);
        assert!(b.is_some());

        let b = b.unwrap();
        check_ball_eq(a, b);

        let children = tree.children_of(depth, index);
        assert_eq!(a.children().len(), children.len());

        for (&c, &(d, _, _)) in a.children().iter().zip(children.iter()) {
            check_ball_eq(c, d);
        }
    }
}

#[test_case(20, 2)]
#[test_case(1_000, 10)]
#[test_case(10_000, 10)]
fn new(car: usize, dim: usize) {
    let max = 1.0;
    let seed = 42;

    let data = common::data_gen::gen_random_data(car, dim, max, seed);
    let metric = Euclidean;
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let tree = Tree::new(&data, &metric, &criteria, Some(seed)).unwrap();
    check_tree(&tree);

    let tree = Tree::par_new(&data, &metric, &criteria, Some(seed)).unwrap();
    check_tree(&tree);
}

fn check_ball_eq(a: &Ball<f64>, b: &Ball<f64>) {
    assert_eq!(a.depth(), b.depth());
    assert_eq!(a.cardinality(), b.cardinality());
    assert_eq!(a.radius(), b.radius());
    assert_eq!(a.lfd(), b.lfd());
    assert_eq!(a.arg_center(), b.arg_center());
    assert_eq!(a.arg_radial(), b.arg_radial());
    assert_eq!(a.indices(), b.indices());
}

fn check_tree(tree: &Tree<f64, Ball<f64>>) {
    let subtree = tree.bft().collect::<Vec<_>>();
    let data_car = tree.root().0.cardinality();
    assert_eq!(data_car + data_car - 1, subtree.len());

    for c in subtree {
        let res = tree.find(c);
        assert!(res.is_some());

        let (depth, index, a, b) = res.unwrap();
        let c_found = tree.get(depth, index);
        assert!(c_found.is_some());

        let c_found = c_found.unwrap();
        check_ball_eq(c, c_found);

        let children = tree.children_of(depth, index);
        if a == b {
            assert!(children.is_empty());
        } else {
            assert_eq!(b - a, children.len());

            let car_sum = children.iter().map(|(c, _, _)| c.cardinality()).sum();
            assert_eq!(c.cardinality(), car_sum);

            for (child, _, _) in children {
                assert!(!child.indices().is_empty());
                assert!(child.is_descendant_of(c));
            }
        }
    }
}
