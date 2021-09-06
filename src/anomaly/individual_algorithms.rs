use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use ndarray::prelude::*;
use num_traits::FromPrimitive;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::core::Subsumed;
use crate::prelude::*;

type ClusterScores<T, U> = Vec<(Arc<Cluster<T, U>>, OrderedFloat<f64>)>;
pub type IndividualAlgorithm<T, U> =
    Box<dyn (Fn(&Arc<Graph<T, U>>) -> Vec<f64>) + Send + Sync>;

pub fn get_individual_algorithms<
    'a,
    T: Number + 'static,
    U: Number + 'static,
>() -> Vec<(&'a str, Arc<IndividualAlgorithm<T, U>>)> {
    vec![
        ("cc", Arc::new(Box::new(cluster_cardinality))),
        ("sc", Arc::new(Box::new(component_cardinality))),
        ("gn", Arc::new(Box::new(graph_neighborhood))),
        ("sp", Arc::new(Box::new(stationary_probabilities))),
        ("cr", Arc::new(Box::new(cardinality_ratio))),
        ("vd", Arc::new(Box::new(vertex_degree))),
    ]
}

fn cluster_cardinality<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> Vec<f64> {
    point_scores(normalize(_cluster_cardinality(graph)))
}

fn component_cardinality<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> Vec<f64> {
    point_scores(normalize(_component_cardinality(graph)))
}

fn graph_neighborhood<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> Vec<f64> {
    point_scores(normalize(_graph_neighborhood(graph)))
}

fn stationary_probabilities<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> Vec<f64> {
    point_scores(normalize(_stationary_probabilities(graph)))
}

fn cardinality_ratio<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> Vec<f64> {
    point_scores(normalize(_cardinality_ratio(graph)))
}

fn vertex_degree<T: Number, U: Number>(graph: &Arc<Graph<T, U>>) -> Vec<f64> {
    point_scores(normalize(_vertex_degree(graph)))
}

fn point_scores<T: Number, U: Number>(scores: ClusterScores<T, U>) -> Vec<f64> {
    let mut scores: Vec<_> = scores
        .into_par_iter()
        .flat_map(|(cluster, score)| {
            cluster
                .indices
                .par_iter()
                .map(|&i| (i, score))
                .collect::<Vec<_>>()
        })
        .collect();

    scores.sort_unstable();
    scores
        .into_iter()
        .map(|(_, score)| f64::from(score))
        .collect()
}

fn normalize<T: Number, U: Number>(
    scores: ClusterScores<T, U>,
) -> ClusterScores<T, U> {
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
        .map(|score| (score - mean) / (std_dev * 2_f64.sqrt()))
        .map(statrs::function::erf::erf)
        .map(|score| (1. + score) / 2.)
        .flat_map(OrderedFloat::from_f64)
        .collect();

    clusters.into_iter().zip(scores.into_iter()).collect()
}

fn _cluster_cardinality<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> ClusterScores<T, U> {
    graph
        .clusters
        .par_iter()
        .map(|cluster| {
            (
                Arc::clone(cluster),
                -OrderedFloat::<f64>::from_usize(cluster.cardinality).unwrap(),
            )
        })
        .collect()
}

fn _component_cardinality<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> ClusterScores<T, U> {
    graph
        .find_components()
        .par_iter()
        .flat_map(|component| {
            let score =
                -OrderedFloat::<f64>::from_usize(component.cardinality).unwrap();
            component
                .clusters
                .par_iter()
                .map(move |cluster| (Arc::clone(cluster), score))
        })
        .collect()
}

fn find_neighborhood_size<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
    start: &Arc<Cluster<T, U>>,
    eccentricity_fraction: f64,
) -> usize {
    let mut steps_to_go = (eccentricity_fraction
        * (graph.eccentricity(start).unwrap() as f64)
        + 1.) as usize;

    let mut visited = HashSet::new();

    let mut frontier = HashSet::new();
    frontier.insert(Arc::clone(start));

    while steps_to_go > 0 && !frontier.is_empty() {
        let new_frontier = frontier
            .par_iter()
            .map(|cluster| {
                graph
                    .neighbors(cluster)
                    .unwrap()
                    .par_iter()
                    .filter(|&neighbor| {
                        !visited.contains(neighbor)
                            && !frontier.contains(neighbor)
                    })
                    .map(Arc::clone)
                    .collect::<Vec<Arc<Cluster<T, U>>>>()
            })
            .flatten()
            .collect();

        visited.extend(frontier.into_iter());
        frontier = new_frontier;
        steps_to_go -= 1;
    }

    visited.len()
}

fn score_subsumed_clusters<T: Number, U: Number>(
    scores: &HashMap<Arc<Cluster<T, U>>, OrderedFloat<f64>>,
    subsumed_neighbors: &Subsumed<T, U>,
) -> ClusterScores<T, U> {
    let subsumed_scores: HashMap<_, _> = subsumed_neighbors
        .par_iter()
        .flat_map(|(master, subsumed)| {
            subsumed.par_iter().map(move |cluster| {
                let score = if scores.contains_key(cluster) {
                    std::cmp::max(
                        *scores.get(cluster).unwrap(),
                        *scores.get(master).unwrap(),
                    )
                } else {
                    *scores.get(master).unwrap()
                };
                (Arc::clone(cluster), score)
            })
        })
        .collect();
    scores
        .par_iter()
        .chain(subsumed_scores.par_iter())
        .map(|(cluster, &score)| (Arc::clone(cluster), score))
        .collect()
}

fn _graph_neighborhood<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> ClusterScores<T, U> {
    let eccentricity_fraction = 0.25;
    let (pruned_graph, subsumed_neighbors) = graph.pruned_graph();

    let scores: HashMap<_, _> = pruned_graph
        .clusters
        .par_iter()
        .map(|cluster| {
            (
                Arc::clone(cluster),
                -OrderedFloat::<f64>::from_usize(find_neighborhood_size(
                    &pruned_graph,
                    cluster,
                    eccentricity_fraction,
                ))
                .unwrap(),
            )
        })
        .collect();

    score_subsumed_clusters(&scores, &subsumed_neighbors)
}

fn steady_matrix<U: Number>(matrix: Array2<U>) -> Array2<f64> {
    let shape = (matrix.nrows(), matrix.ncols());
    let matrix: Array1<f64> = matrix
        .outer_iter()
        .flat_map(|row| {
            let sum = row.sum().to_f64();
            row.into_iter().map(move |&val| val.to_f64() / sum)
        })
        .collect();
    let mut matrix: Array2<f64> = matrix.into_shape(shape).unwrap();

    // TODO: Go until convergence
    let steps = 16;
    for _ in 0..steps {
        matrix = matrix.dot(&matrix);
    }

    matrix
}

fn _stationary_probabilities<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> ClusterScores<T, U> {
    let (pruned_graph, subsumed_neighbors) = graph.pruned_graph();

    let scores: HashMap<_, _> = pruned_graph
        .find_components()
        .iter()
        .flat_map(|component| {
            let scores: ClusterScores<T, U> = if component.cardinality > 1 {
                let (clusters, matrix) = component.distance_matrix();
                let sums = steady_matrix(matrix).sum_axis(Axis(0)).to_vec();
                clusters
                    .into_iter()
                    .zip(sums.into_iter())
                    .map(|(cluster, score)| {
                        (cluster, -OrderedFloat::<f64>::from_f64(score).unwrap())
                    })
                    .collect()
            } else {
                component
                    .clusters
                    .iter()
                    .map(|cluster| {
                        (
                            Arc::clone(cluster),
                            OrderedFloat::from_f64(1_f64).unwrap(),
                        )
                    })
                    .collect()
            };
            scores.into_iter()
        })
        .collect();

    score_subsumed_clusters(&scores, &subsumed_neighbors)
}

fn _cardinality_ratio<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> ClusterScores<T, U> {
    let weight = |i: usize| 1. / (i as f64).sqrt();
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
            (Arc::clone(cluster), OrderedFloat::from_f64(score).unwrap())
        })
        .collect()
}

fn _vertex_degree<T: Number, U: Number>(
    graph: &Arc<Graph<T, U>>,
) -> ClusterScores<T, U> {
    let (clusters, matrix) = graph.adjacency_matrix();
    let scores: Vec<_> = matrix
        .outer_iter()
        .into_par_iter()
        .map(|row| {
            -OrderedFloat::<f64>::from_usize(row.into_iter().count()).unwrap()
        })
        .collect();
    clusters.into_iter().zip(scores.into_iter()).collect()
}
