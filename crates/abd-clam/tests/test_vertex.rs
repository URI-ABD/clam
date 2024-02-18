use abd_clam::{graph::Vertex, Cluster, PartitionCriteria, Tree};
use abd_clam::{Dataset, Instance, VecDataset};
use distances::Number;
use tempdir::TempDir;

mod utils;

type Vertexf32 = Vertex<f32>;
type DataSetf32 = VecDataset<Vec<f32>, f32, usize>;
type Treef32 = Tree<Vec<f32>, f32, DataSetf32, Vertexf32>;

#[test]
fn save_load_vertex() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);
    let metric = data.metric();

    let criteria = PartitionCriteria::default();
    let raw_tree = Treef32::new(data, Some(42))
        .partition(&criteria, Some(42))
        .normalize_ratios();

    let tree_dir = TempDir::new("tree_medium_vertex").unwrap();

    // Save the tree
    raw_tree.save(tree_dir.path()).unwrap();

    // Recover the tree
    let rec_tree = Treef32::load(tree_dir.path(), metric, false).unwrap();

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
fn assert_subtree_equal<I: Instance, U: Number, M: Instance>(
    raw_cluster: &Vertex<U>,
    raw_data: &VecDataset<I, U, M>,
    rec_cluster: &Vertex<U>,
    rec_data: &VecDataset<I, U, M>,
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

    // assert ratios are equal
    for (r1, r2) in raw_cluster.ratios().iter().zip(rec_cluster.ratios()) {
        float_cmp::assert_approx_eq!(f64, *r1, r2, epsilon = 0.00000003);
    }

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
fn ratios() {
    // Generate some tree from a small dataset
    let data = utils::gen_dataset_from(
        vec![vec![10.], vec![1.], vec![3.]],
        utils::euclidean::<f32, f32>,
        vec![true, true, false],
    );

    let partition_criteria = PartitionCriteria::default();
    let root = Tree::<_, _, _, Vertex<_>>::new(data, Some(42)).partition(&partition_criteria, None);

    //          1 (root)
    //    10 (lc)       11 (rc)
    // 100 (lclc)  101 (lcrc)

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
        all_cardinalities[4] as f64 / root_ratios[0] as f64,
        all_radius[4] as f64 / root_ratios[1] as f64,
        all_lfd[4] / root_ratios[2],
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
