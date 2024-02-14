//! Tests for the `Vertex` struct.

use abd_clam::{graph::Vertex, Cluster, PartitionCriteria, Tree};

mod utils;

#[test]
fn ratios() {
    // Generate some tree from a small dataset
    let data = utils::gen_dataset_from(
        vec![vec![10.], vec![1.], vec![3.]],
        utils::euclidean::<f32, f32>,
        vec![true, true, false],
    );

    let partition_criteria = PartitionCriteria::default();
    let root = Tree::<_, _, _, Vertex<_>>::new(data, Some(42)).partition(&partition_criteria, None);

    //          1
    //    10        11
    // 100  101

    let all_ratios = root
        .root()
        .subtree()
        .into_iter()
        .map(|c| c.ratios())
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
    let raw_tree = Tree::<_, _, _, Vertex<_>>::new(data, None)
        .partition(&partition_criteria, None)
        .normalize_ratios();

    let all_ratios = raw_tree
        .root()
        .subtree()
        .into_iter()
        .map(|c| c.ratios())
        .collect::<Vec<_>>();

    for row in &all_ratios {
        for val in row {
            assert!(*val >= 0. && *val <= 1.);
        }
    }
}
