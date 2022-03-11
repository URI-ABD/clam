//! These are the individual algorithms used in the ensemble for CHAODA.
//! All algorithms operate on a graph and produce anomaly rankings for all instances in that graph.
//! Each algorithm contributes a different inductive bias with a different approach to ranking anomalies.
//! See the paper for details on each algorithm.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use ndarray::prelude::*;
use rayon::prelude::*;

use crate::core::Subsumed;
use crate::prelude::*;

// TODO: Create Struct for IndividualAlgorithm. Add decision methods for each to intelligently decide when to apply the algorithm.
// This should be used to handle speed thresholds and also to avoid the case where all clusters would be assigned the same score.
// pub type IndividualAlgorithm<T, U> = Box<dyn () + Send + Sync>;
type ClusterScores<T, U> = Vec<(Arc<Cluster<T, U>>, f64)>;

/// A `Box`ed function that takes a graph and retuns anomaly rankings of all instances in that graph.
pub type IndividualAlgorithm<T, U> = Box<fn(Arc<Graph<T, U>>) -> Vec<f64>>;

pub fn get_individual_algorithms<T: Number, U: Number>() -> Vec<(String, Arc<IndividualAlgorithm<T, U>>)> {
    vec![
        ("sc".to_string(), Arc::new(Box::new(subgraph_cardinality))),
        ("cc".to_string(), Arc::new(Box::new(cluster_cardinality))),
        ("gn".to_string(), Arc::new(Box::new(graph_neighborhood))),
        ("sp".to_string(), Arc::new(Box::new(stationary_probabilities))),
        ("cr".to_string(), Arc::new(Box::new(cardinality_ratio))),
        ("vd".to_string(), Arc::new(Box::new(vertex_degree))),
    ]
}

/// Relative Cluster Cardinality
fn cluster_cardinality<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> Vec<f64> {
    get_instance_scores(normalize(_cluster_cardinality(graph)))
}

/// Relative Component Cardinality
fn subgraph_cardinality<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> Vec<f64> {
    get_instance_scores(normalize(_subgraph_cardinality(graph)))
}

/// Graph Neighborhood Size
fn graph_neighborhood<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> Vec<f64> {
    get_instance_scores(normalize(_graph_neighborhood(graph)))
}

/// Stationary Probabilities
fn stationary_probabilities<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> Vec<f64> {
    get_instance_scores(normalize(_stationary_probabilities(graph)))
}

/// Child-parent Cardinality Ratios
fn cardinality_ratio<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> Vec<f64> {
    get_instance_scores(normalize(_cardinality_ratio(graph)))
}

/// Relative Vertex Degree
fn vertex_degree<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> Vec<f64> {
    get_instance_scores(normalize(_vertex_degree(graph)))
}

/// Given scores for clusters, get scored for the individual instances.
fn get_instance_scores<T: Number, U: Number>(scores: ClusterScores<T, U>) -> Vec<f64> {
    let mut scores: Vec<_> = scores
        .into_par_iter()
        .flat_map(|(cluster, score)| cluster.indices.par_iter().map(|&i| (i, score)).collect::<Vec<_>>())
        .collect();

    scores.sort_by(|a, b| a.0.cmp(&b.0));
    let (_, scores): (Vec<_>, Vec<_>) = scores.into_iter().unzip();
    scores
}

/// Cormalize raw cluster scores to a [0, 1] range, using gaussian normalization.
fn normalize<T: Number, U: Number>(scores: ClusterScores<T, U>) -> ClusterScores<T, U> {
    let (clusters, scores): (Vec<_>, Vec<_>) = scores.into_iter().unzip();
    let scores: Vec<_> = scores.into_iter().map(f64::from).collect();

    let num_scores = scores.len() as f64;

    let mean = scores.iter().sum::<f64>() / num_scores;
    let std_dev = 1e-8
        + scores
            .par_iter()
            .map(|&score| score - mean)
            .map(|difference| difference.powi(2))
            .sum::<f64>()
            .sqrt()
            / num_scores;

    let scores: Vec<_> = scores
        .into_par_iter()
        .map(|score| (score - mean) / (std_dev * 2_f64.sqrt())) // look into removing the sqrt(2) factor
        .map(statrs::function::erf::erf)
        .map(|score| (1. + score) / 2.)
        .collect();

    clusters.into_iter().zip(scores.into_iter()).collect()
}

fn _cluster_cardinality<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> ClusterScores<T, U> {
    graph
        .clusters
        .par_iter()
        .map(|cluster| (Arc::clone(cluster), -(cluster.cardinality as f64)))
        .collect()
}

fn _subgraph_cardinality<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> ClusterScores<T, U> {
    graph
        .find_components()
        .par_iter()
        .flat_map(|component| {
            component
                .clusters
                .par_iter()
                .map(|cluster| (Arc::clone(cluster), -(component.cardinality as f64)))
        })
        .collect()
}

/// Returns the number of clusters reachable from the starting cluster within a number of steps equal to the given fraction of the eccentricity of the cluster.
fn find_neighborhood_size<T: Number, U: Number>(
    graph: Arc<Graph<T, U>>,
    start: Arc<Cluster<T, U>>,
    eccentricity_fraction: f64,
) -> usize {
    let steps_to_go = (eccentricity_fraction * (graph.eccentricity(&start).unwrap() as f64) + 1.) as usize;

    let mut visited = HashSet::new();

    let mut frontier = HashSet::new();
    frontier.insert(Arc::clone(&start));

    for _ in 0..steps_to_go {
        if frontier.is_empty() {
            break;
        } else {
            visited.extend(frontier.iter().cloned());

            frontier = frontier
                .par_iter()
                .flat_map(|cluster| {
                    graph
                        .neighbors(cluster)
                        .unwrap()
                        .par_iter()
                        .filter(|&neighbor| !visited.contains(neighbor))
                        .map(Arc::clone)
                        .collect::<Vec<_>>()
                })
                .collect();
        }
    }

    visited.len()
}

/// Compute scores for subsumed clusters given scores for the subsumer clusters.
fn score_subsumed_clusters<T: Number, U: Number>(
    scores: HashMap<Arc<Cluster<T, U>>, f64>,
    subsumed_neighbors: Subsumed<T, U>,
) -> ClusterScores<T, U> {
    let subsumed_scores: HashMap<_, _> = subsumed_neighbors
        .par_iter()
        .flat_map(|(master, subsumed)| {
            subsumed.par_iter().map(|cluster| {
                let score = if scores.contains_key(cluster) {
                    let (s1, s2) = (*scores.get(cluster).unwrap(), *scores.get(master).unwrap());
                    if s1 > s2 {
                        s1
                    } else {
                        s2
                    }
                } else {
                    *scores.get(master).unwrap()
                };
                (Arc::clone(cluster), score)
            })
        })
        .collect();

    scores.into_iter().chain(subsumed_scores.into_iter()).collect()
}

fn _graph_neighborhood<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> ClusterScores<T, U> {
    let eccentricity_fraction = 0.25;
    let (pruned_graph, subsumed_neighbors) = graph.pruned_graph();

    let scores: HashMap<_, _> = pruned_graph
        .clusters
        .par_iter()
        .map(|cluster| {
            let score = find_neighborhood_size(Arc::clone(&pruned_graph), Arc::clone(cluster), eccentricity_fraction);
            (Arc::clone(cluster), -(score as f64))
        })
        .collect();

    score_subsumed_clusters(scores, subsumed_neighbors)
}

/// Returns the square matrix of stationary probabilities.
fn steady_matrix<U: Number>(matrix: Array2<U>) -> Array2<f64> {
    let shape = (matrix.nrows(), matrix.ncols());
    let matrix: Array1<f64> = matrix
        .outer_iter()
        .flat_map(|row| {
            let sum = row.sum().as_f64();
            row.into_iter().map(move |&val| val.as_f64() / sum)
        })
        .collect();
    let mut matrix: Array2<f64> = matrix.into_shape(shape).unwrap();

    // TODO Go until convergence
    let steps = 16;
    for _ in 0..steps {
        matrix = matrix.dot(&matrix);
    }

    matrix
}

fn _stationary_probabilities<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> ClusterScores<T, U> {
    let (pruned_graph, subsumed_neighbors) = graph.pruned_graph();

    let scores: HashMap<_, _> = pruned_graph
        .find_components()
        .iter()
        .flat_map(|component| {
            let scores: ClusterScores<T, U> = if component.cardinality > 1 {
                let (clusters, matrix) = component.distance_matrix();
                let sums = steady_matrix(matrix).sum_axis(Axis(0)).to_vec();
                assert_eq!(
                    sums.len(),
                    clusters.len(),
                    "sums and clusters did not have equal length"
                );
                clusters
                    .into_iter()
                    .zip(sums.into_iter())
                    .map(|(cluster, score)| (cluster, -score))
                    .collect()
            } else {
                component
                    .clusters
                    .iter()
                    .map(|cluster| (Arc::clone(cluster), 0_f64))
                    .collect()
            };
            scores.into_iter()
        })
        .collect();

    score_subsumed_clusters(scores, subsumed_neighbors)
}

fn _cardinality_ratio<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> ClusterScores<T, U> {
    let weight = |i: usize| 1. / (i as f64 * 0.5);
    graph
        .clusters
        .par_iter()
        .map(|cluster| {
            let ancestors = cluster.ancestry();
            let score = ancestors
                .iter()
                .skip(1)
                .chain([Arc::clone(cluster)].iter())
                .map(|cluster| cluster.cardinality as f64)
                .zip(ancestors.iter().map(|cluster| cluster.cardinality as f64))
                .enumerate()
                .map(|(i, (c, p))| weight(i + 1) * c / p)
                .sum::<f64>();
            (Arc::clone(cluster), -score)
        })
        .collect()
}

fn _vertex_degree<T: Number, U: Number>(graph: Arc<Graph<T, U>>) -> ClusterScores<T, U> {
    let (clusters, matrix) = graph.adjacency_matrix();
    let scores: Vec<_> = matrix
        .outer_iter()
        .into_par_iter()
        .map(|row| -(row.into_iter().filter(|&&b| b).count() as f64))
        .collect();
    clusters.into_iter().zip(scores.into_iter()).collect()
}
