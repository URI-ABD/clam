//! Checking properties of clusters.

use abd_clam::{Ball, Cluster, Dataset, FlatVec};

pub fn test_new(root: &Ball<i32>, data: &FlatVec<Vec<i32>, usize>) {
    let arg_r = root.arg_radial();
    let indices = (0..data.cardinality()).collect::<Vec<_>>();

    assert_eq!(arg_r, data.cardinality() - 1);
    assert_eq!(root.depth(), 0);
    assert_eq!(root.cardinality(), 5);
    assert_eq!(root.arg_center(), 2);
    assert_eq!(root.radius(), 12);
    assert_eq!(root.arg_radial(), arg_r);
    assert!(root.children().is_empty());
    assert_eq!(root.indices(), indices);
    assert_eq!(root.extents().len(), 1);
}

pub fn check_partition(root: &Ball<i32>) -> bool {
    let indices = root.indices();

    assert!(!root.children().is_empty());
    assert_eq!(indices, &[0, 1, 2, 4, 3]);
    assert_eq!(root.extents().len(), 2);

    let children = root.children();
    assert_eq!(children.len(), 2);
    for &c in &children {
        assert_eq!(c.depth(), 1);
        assert!(c.children().is_empty());
    }

    let (left, right) = (children[0], children[1]);

    assert_eq!(left.cardinality(), 3);
    assert_eq!(left.arg_center(), 1);
    assert_eq!(left.radius(), 4);
    assert!([0, 2].contains(&left.arg_radial()));

    assert_eq!(right.cardinality(), 2);
    assert_eq!(right.radius(), 8);
    assert!([3, 4].contains(&right.arg_center()));
    assert!([3, 4].contains(&right.arg_radial()));

    true
}
