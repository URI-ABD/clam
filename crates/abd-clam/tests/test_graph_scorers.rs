use abd_clam::chaoda::graph_scorers::{ClusterCardinality, ComponentCardinality, GraphScorer, VertexDegree};
use abd_clam::chaoda::pretrained_models;
use abd_clam::{Cluster, Graph, PartitionCriteria, Tree, VecDataset};
use distances::number::Float;
use distances::Number;
use rand::SeedableRng;
use smartcore::linalg::BaseVector;

/// Generate a dataset with the given cardinality and dimensionality.
pub fn gen_dataset_with_anomaly(
    cardinality: usize,
    dimensionality: usize,
    seed: u64,
    metric: fn(&Vec<f32>, &Vec<f32>) -> f32,
    anomalies: usize,
) -> VecDataset<Vec<f32>, f32, usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = symagen::random_data::random_tabular(cardinality - anomalies, dimensionality, -0.01, 0.01, &mut rng);
    for i in 0..anomalies {
        data.push(vec![10000.; dimensionality]);
    }
    let name = "test".to_string();
    VecDataset::new(name, data, metric, false)
}

/// Euclidean distance between two vectors.
pub fn euclidean<T: Number, F: Float>(x: &Vec<T>, y: &Vec<T>) -> F {
    distances::vectors::euclidean(x, y)
}

#[test]
fn test_score_graph_basics() {
    let data = gen_dataset_with_anomaly(1000, 10, 42, euclidean, 1);
    let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
    let raw_tree = Tree::new(data, Some(42))
        .partition(&partition_criteria)
        .with_ratios(true);

    let graph = Graph::from_tree(
        &raw_tree,
        &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
        4,
    )
    .unwrap();
    let scorer = VertexDegree;
    let mut results = scorer.call(&graph).unwrap();

    assert_eq!(results.0.len(), graph.clusters().len());
    assert_eq!(results.1.len(), graph.population());
}

#[test]
fn test_vertex_scorer() {
    for anomaly_count in 0..5 {
        let data = gen_dataset_with_anomaly(1000, 10, 42, euclidean, anomaly_count);
        let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
        let raw_tree = Tree::new(data, Some(42))
            .partition(&partition_criteria)
            .with_ratios(true);

        let graph = Graph::from_tree(
            &raw_tree,
            &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
            4,
        )
        .unwrap();
        let scorer = VertexDegree;
        let mut results = scorer.call(&graph).unwrap();

        let mut mean = results.1.sum() / results.1.len().as_f64();
        let mut means_sqr = results.1.iter().map(|score| score.powi(2)).sum::<f64>() / results.1.len().as_f64();
        let standard_dev = (means_sqr - mean.powi(2)).sqrt();

        let outliers: Vec<f64> = results
            .1
            .iter()
            .filter(|&&x| (x - mean).abs() > 5. * standard_dev)
            .cloned()
            .collect();

        assert_eq!(outliers.len(), anomaly_count);
        if anomaly_count > 0 {
            outliers.iter().for_each(|o| assert!(o > &mean));
        }
    }
}

#[test]
fn test_component_scorer() {
    for anomaly_count in 0..5 {
        let data = gen_dataset_with_anomaly(1000, 10, 42, euclidean, anomaly_count);
        let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
        let raw_tree = Tree::new(data, Some(42))
            .partition(&partition_criteria)
            .with_ratios(true);

        let graph = Graph::from_tree(
            &raw_tree,
            &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
            4,
        )
        .unwrap();
        let scorer = ComponentCardinality;
        let mut results = scorer.call(&graph).unwrap();

        let mut mean = results.1.sum() / results.1.len().as_f64();
        let mut means_sqr = results.1.iter().map(|score| score.powi(2)).sum::<f64>() / results.1.len().as_f64();
        let standard_dev = (means_sqr - mean.powi(2)).sqrt();

        let outliers: Vec<f64> = results
            .1
            .iter()
            .filter(|&&x| (x - mean).abs() > 5. * standard_dev)
            .cloned()
            .collect();

        assert_eq!(outliers.len(), anomaly_count);
        if anomaly_count > 0 {
            outliers.iter().for_each(|o| assert!(o > &mean));
        }
    }
}

#[test]
fn test_cluster_scorer() {
    let anomaly_count = 1;
    let data = gen_dataset_with_anomaly(10000, 10, 42, euclidean, anomaly_count);
    let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
    let raw_tree = Tree::new(data, Some(42))
        .partition(&partition_criteria)
        .with_ratios(true);

    let graph = Graph::from_tree(
        &raw_tree,
        &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
        4,
    )
    .unwrap();

    let scorer = ClusterCardinality;
    let call_results = scorer.call(&graph).unwrap();
    let mut highest_score: Option<(&Cluster<f32>, f64)> = None;
    for i in call_results.0 {
        if highest_score == None {
            highest_score = Some(i)
        } else if highest_score.unwrap().1 < i.1 {
            highest_score = Some(i)
        }
    }

    // 1 anomaly inserted at end [9999] with dataset generation
    assert!(highest_score.unwrap().0.indices().contains(&9999))
}
