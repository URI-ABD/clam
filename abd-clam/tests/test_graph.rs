use abd_clam::{EdgeSet, Graph, PartitionCriteria, Tree};

use abd_clam::builder::detect_edges;
use abd_clam::cluster_selection::select_clusters;

mod utils;

#[test]
fn create_graph() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);
    let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
    let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria);
    let selected_clusters = select_clusters(raw_tree.root(), String::from("lr_euclidean_cc")).unwrap();

    let edges = detect_edges(&selected_clusters, raw_tree.data());

    let graph = Graph::new(selected_clusters.clone(), edges.clone());

    if let Ok(graph) = graph {
        // assert edges and clusters are correct
        assert_eq!(graph.clusters().len(), selected_clusters.len());
        assert_eq!(graph.edges().len(), edges.len());

        let reference_population = selected_clusters.iter().fold(0, |acc, &c| acc + c.cardinality());
        assert_eq!(graph.population(), reference_population);
        let components = graph.find_component_clusters();

        // assert ordered clusters are in correct order
        graph
            .clusters()
            .iter()
            .zip(graph.ordered_clusters().iter())
            .for_each(|(c1, c2)| {
                assert_eq!(c1, c2);
            });

        let num_clusters_in_components = components.iter().map(|c| c.len()).sum::<usize>();
        assert_eq!(num_clusters_in_components, selected_clusters.len());

        // assert the number of clusters in a component is equal to the number of clusters in each of its cluster's traversals
        for component in &components {
            for c in component {
                if let Ok(traversal_result) = graph.traverse(c) {
                    assert_eq!(traversal_result.0.len(), component.len());
                    // assert_eq!(traversal_result.1.len(), component.len());
                }
            }
        }
    }
}

#[test]
fn adjacency_map() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);
    let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
    let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria);
    let selected_clusters = select_clusters(raw_tree.root(), "lr_euclidean_cc".to_string()).unwrap();

    let edges = detect_edges(&selected_clusters, raw_tree.data());

    if let Ok(graph) = Graph::new(selected_clusters.clone(), edges.clone()) {
        let adj_map = graph.adjacency_map();
        assert_eq!(adj_map.len(), graph.clusters().len());

        for component in &graph.find_component_clusters() {
            for c in component {
                let adj = adj_map.get(c).unwrap();
                for adj_c in adj {
                    assert!(component.contains(adj_c));
                }
            }
        }
    }
}
