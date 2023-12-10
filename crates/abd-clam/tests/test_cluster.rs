//! Tests for the `Cluster` struct.

use abd_clam::{Cluster, Dataset, PartitionCriteria, Tree, VecDataset};

mod utils;

#[test]
fn tiny() {
    let mut data = utils::gen_dataset_from(
        vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]],
        utils::euclidean::<f32, f32>,
        Some(vec![true, true, false, false]),
    );
    let partition_criteria = PartitionCriteria::default();
    let root = Cluster::new_root(&data, Some(42)).partition(&mut data, &partition_criteria);

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
    let root = Cluster::new_root(&data, None).partition(&mut data, &partition_criteria);

    assert!(!root.is_leaf());
    assert!(root.children().is_some());
    assert_eq!(root.depth(), 0);
    assert_eq!(root.cardinality(), 10_000);

    check_subtree(&root, &data);
}

fn check_subtree(root: &Cluster<f32>, data: &VecDataset<Vec<f32>, f32, bool>) {
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
        Some(vec![true, true, false, false]),
    );

    let original = Cluster::new_root(&data, Some(42));
    // original.history = vec![true, true, false, false, true];

    let original_bytes = postcard::to_allocvec(&original).unwrap();
    let deserialized: Cluster<f32> = postcard::from_bytes(&original_bytes).unwrap();

    assert_eq!(original.name(), deserialized.name());
    assert_eq!(original.cardinality(), deserialized.cardinality());
    assert_eq!(original.indices(), deserialized.indices());
    assert_eq!(original.arg_center(), deserialized.arg_center());
    assert_eq!(original.arg_radial(), deserialized.arg_radial());
    assert_eq!(original.lfd(), deserialized.lfd());
    assert_eq!(original.ratios(), deserialized.ratios());
    assert_eq!(original.depth(), deserialized.depth());
    assert_eq!(original.radius(), deserialized.radius());
    assert_eq!(original.children(), deserialized.children());
}

#[test]
fn ratios() {
    // Generate some tree from a small dataset
    let data = utils::gen_dataset_from(vec![vec![10.], vec![1.], vec![3.]], utils::euclidean::<f32, f32>, None);

    let partition_criteria = PartitionCriteria::default();
    let root = Tree::new(data, Some(42))
        .partition(&partition_criteria)
        .with_ratios(false);

    //          1
    //    10        11
    // 100  101

    let all_ratios = root
        .root()
        .subtree()
        .into_iter()
        .map(|c| c.ratios().unwrap())
        .collect::<Vec<_>>();

    let all_cardinalities = root
        .root()
        .subtree()
        .into_iter()
        .map(|c| c.cardinality())
        .collect::<Vec<_>>();

    let all_lfd = root.root().subtree().into_iter().map(|c| c.lfd()).collect::<Vec<_>>();

    let all_radius = root
        .root()
        .subtree()
        .into_iter()
        .map(|c| c.radius())
        .collect::<Vec<_>>();

    // manually calculate ratios between root and its children
    let root_ratios = vec![
        all_cardinalities[0] as f64,
        all_radius[0] as f64,
        all_lfd[0],
        -1.,
        -1.,
        -1.,
    ];
    let lc_ratios = vec![
        all_cardinalities[1] as f64 / root_ratios[0] as f64,
        all_radius[1] as f64 / root_ratios[1] as f64,
        all_lfd[1] / root_ratios[2],
        -1.,
        -1.,
        -1.,
    ];
    let lclc_ratios = vec![
        all_cardinalities[2] as f64 / lc_ratios[0] as f64,
        all_radius[2] as f64 / lc_ratios[1] as f64,
        all_lfd[2] / lc_ratios[2],
        -1.,
        -1.,
        -1.,
    ];
    let lcrc_ratios = vec![
        all_cardinalities[3] as f64 / lc_ratios[0] as f64,
        all_radius[3] as f64 / lc_ratios[1] as f64,
        all_lfd[3] / lc_ratios[2],
        -1.,
        -1.,
        -1.,
    ];
    let rc_ratios = vec![
        all_cardinalities[3] as f64 / root_ratios[0] as f64,
        all_radius[3] as f64 / root_ratios[1] as f64,
        all_lfd[3] / root_ratios[2],
        -1.,
        -1.,
        -1.,
    ];

    assert_eq!(all_ratios[0][..3], root_ratios[..3], "root not correct");
    assert_eq!(all_ratios[1][..3], lc_ratios[..3], "lc not correct");
    assert_eq!(all_ratios[2][..3], lclc_ratios[..3], "lclc not correct");
    assert_eq!(all_ratios[3][..3], lcrc_ratios[..3], "lcrc not correct");
    assert_eq!(all_ratios[4][..3], rc_ratios[..3], "rc not correct");
}

#[test]
fn normalized_ratios() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);

    let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
    let raw_tree = Tree::new(data, None).partition(&partition_criteria).with_ratios(true);

    let all_ratios = raw_tree
        .root()
        .subtree()
        .into_iter()
        .map(|c| c.ratios().unwrap())
        .collect::<Vec<_>>();

    for row in &all_ratios {
        for val in row {
            assert!(*val >= 0. && *val <= 1.);
        }
    }
}
