//! Tests for the `UniBall` struct.

use abd_clam::{Cluster, Dataset, Instance, PartitionCriteria, UniBall, VecDataset};

mod utils;

#[test]
fn tiny() {
    let mut data = utils::gen_dataset_from(
        vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]],
        utils::euclidean::<f32, f32>,
        vec![true, true, false, false],
    );
    let partition_criteria = PartitionCriteria::default();
    let root = UniBall::new_root(&data, Some(42)).partition(&mut data, &partition_criteria, Some(42));

    assert!(!root.is_leaf());
    assert!(root.children().is_some());

    assert_eq!(root.depth(), 0);
    assert_eq!(root.cardinality(), 4);
    assert_eq!(root.subtree().len(), 7);
    assert!(root.radius() > 0.);

    assert_eq!(format!("{root}"), "0-4");

    let Some([left, right]) = root.children() else {
        unreachable!("The root cluster has children.")
    };
    assert_eq!(format!("{left}"), "0-2");
    assert_eq!(format!("{right}"), "2-2");

    for child in [left, right] {
        assert_eq!(child.depth(), 1);
        assert_eq!(child.cardinality(), 2);
        assert_eq!(child.subtree().len(), 3);
    }

    let subtree = root.subtree();
    assert_eq!(
        subtree.len(),
        7,
        "The subtree of the root cluster should have 7 elements but had {}.",
        subtree.len()
    );

    check_subtree(&root, &data);
}

#[test]
fn medium() {
    let mut data = utils::gen_dataset(10_000, 10, 42, utils::euclidean);
    let partition_criteria = PartitionCriteria::default();
    let root = UniBall::new_root(&data, None).partition(&mut data, &partition_criteria, None);

    assert!(!root.is_leaf());
    assert!(root.children().is_some());
    assert_eq!(root.depth(), 0);
    assert_eq!(root.cardinality(), 10_000);

    check_subtree(&root, &data);
}

fn check_subtree<M: Instance, C: Cluster<f32>>(root: &C, data: &VecDataset<Vec<f32>, f32, M>) {
    for c in root.subtree() {
        assert!(c.cardinality() > 0, "Cardinality must be positive.");
        assert!(c.radius() >= 0., "Radius must be non-negative.");
        assert!(c.lfd() > 0., "LFD must be positive.");

        let radius = data.one_to_one(c.arg_center(), c.arg_radial());
        assert!(
            (radius - c.radius()).abs() <= f32::EPSILON,
            "Radius must be equal to the distance to the farthest instance. {c} had radius {} but distance {radius}.",
            c.radius(),
        );
    }
}

#[test]
fn serialization() {
    let data = utils::gen_dataset_from(
        vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]],
        utils::euclidean::<f32, f32>,
        vec![true, true, false, false],
    );

    let original = UniBall::new_root(&data, Some(42));
    // original.history = vec![true, true, false, false, true];

    let original_bytes = bincode::serialize(&original).unwrap();
    let deserialized: UniBall<f32> = bincode::deserialize(&original_bytes).unwrap();

    assert_eq!(original.name(), deserialized.name());
    assert_eq!(original.cardinality(), deserialized.cardinality());
    assert_eq!(original.indices(), deserialized.indices());
    assert_eq!(original.arg_center(), deserialized.arg_center());
    assert_eq!(original.arg_radial(), deserialized.arg_radial());
    assert_eq!(original.lfd(), deserialized.lfd());
    assert_eq!(original.depth(), deserialized.depth());
    assert_eq!(original.radius(), deserialized.radius());
    assert_eq!(original.children(), deserialized.children());
}
