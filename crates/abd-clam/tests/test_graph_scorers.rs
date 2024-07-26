// use abd_clam::{
//     chaoda::{
//         graph_scorers::{ClusterCardinality, ComponentCardinality, GraphScorer, VertexDegree},
//         pretrained_models, Vertex,
//     },
//     graph::Graph,
//     utils::{mean, standard_deviation},
//     Cluster, PartitionCriteria, Tree, VecDataset,
// };
// use distances::{number::Float, Number};
// use rand::SeedableRng;

// /// Generate a dataset with the given cardinality and dimensionality.
// pub fn gen_dataset_with_anomaly(
//     cardinality: usize,
//     dimensionality: usize,
//     seed: u64,
//     metric: fn(&Vec<f32>, &Vec<f32>) -> f32,
//     anomalies: usize,
// ) -> VecDataset<Vec<f32>, f32, usize> {
//     let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
//     let mut data = symagen::random_data::random_tabular(cardinality - anomalies, dimensionality, -0.01, 0.01, &mut rng);
//     for _ in 0..anomalies {
//         data.push(vec![10000.; dimensionality]);
//     }
//     let name = "test".to_string();
//     VecDataset::new(name, data, metric, false)
// }

// /// Euclidean distance between two vectors.
// #[allow(clippy::ptr_arg)]
// pub fn euclidean<T: Number, F: Float>(x: &Vec<T>, y: &Vec<T>) -> F {
//     distances::vectors::euclidean(x, y)
// }

// #[test]
// fn test_score_graph_basics() {
//     let data = gen_dataset_with_anomaly(1000, 10, 42, euclidean, 1);
//     let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
//     let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria, Some(42));

//     let graph = Graph::from_tree(
//         &raw_tree,
//         &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
//         4,
//     )
//     .unwrap();
//     let scorer = VertexDegree;
//     let results = scorer.call(&graph).unwrap();

//     assert_eq!(results.0.len(), graph.ordered_clusters().len());
//     assert_eq!(results.1.len(), graph.population());
// }

// #[test]
// fn test_vertex_scorer() {
//     for anomaly_count in 0..5 {
//         let data = gen_dataset_with_anomaly(1000, 10, 42, euclidean, anomaly_count);
//         let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
//         let raw_tree = Tree::new(data, Some(42))
//             .partition(&partition_criteria, Some(42))
//             .normalize_ratios();

//         let graph = Graph::from_tree(
//             &raw_tree,
//             &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
//             4,
//         )
//         .unwrap();
//         let scorer = VertexDegree;
//         let (_, scores) = scorer.call(&graph).unwrap();

//         let mean = mean(&scores);
//         let standard_dev = standard_deviation(&scores);

//         let outliers = scores
//             .iter()
//             .filter(|&x| (x - mean).abs() > 5. * standard_dev)
//             .copied()
//             .collect::<Vec<_>>();

//         assert_eq!(outliers.len(), anomaly_count);
//         if anomaly_count > 0 {
//             outliers.iter().for_each(|&o| assert!(o > mean));
//         }
//     }
// }

// #[test]
// fn test_component_scorer() {
//     for anomaly_count in 0..5 {
//         let data = gen_dataset_with_anomaly(1000, 10, 42, euclidean, anomaly_count);
//         let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
//         let raw_tree = Tree::new(data, Some(42))
//             .partition(&partition_criteria, Some(42))
//             .normalize_ratios();

//         let graph = Graph::from_tree(
//             &raw_tree,
//             &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
//             4,
//         )
//         .unwrap();
//         let scorer = ComponentCardinality;
//         let (_, scores) = scorer.call(&graph).unwrap();

//         let mean = mean(&scores);
//         let standard_dev = standard_deviation(&scores);

//         let outliers = scores
//             .iter()
//             .filter(|&x| (x - mean).abs() > 5.0 * standard_dev)
//             .copied()
//             .collect::<Vec<_>>();

//         assert_eq!(outliers.len(), anomaly_count);
//         if anomaly_count > 0 {
//             outliers.iter().for_each(|&o| assert!(o > mean));
//         }
//     }
// }

// #[test]
// fn test_cluster_scorer() {
//     let anomaly_count = 1;
//     let data = gen_dataset_with_anomaly(10000, 10, 42, euclidean, anomaly_count);
//     let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
//     let raw_tree = Tree::new(data, Some(42))
//         .partition(&partition_criteria, Some(42))
//         .normalize_ratios();

//     let graph = Graph::from_tree(
//         &raw_tree,
//         &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
//         4,
//     )
//     .unwrap();

//     let scorer = ClusterCardinality;
//     let call_results = scorer.call(&graph).unwrap();
//     let mut highest_score: Option<(&Vertex<f32>, f64)> = None;
//     for i in call_results.0 {
//         if highest_score.is_none() || highest_score.unwrap().1 < i.1 {
//             highest_score = Some(i)
//         }
//     }

//     // 1 anomaly inserted at end [9999] with dataset generation
//     assert!(highest_score.unwrap().0.indices().contains(&9999))
// }
