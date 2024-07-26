//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms (CHAODA)

mod cluster;
mod component;
mod graph;
mod members;
mod meta_ml;

use std::path::Path;

pub use cluster::{OddBall, Ratios, Vertex};
pub use component::Component;
pub use graph::Graph;
pub use members::Member;
pub use meta_ml::MlModel;

use distances::Number;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::metrics::roc_auc_score;

use crate::{Dataset, Instance, PartitionCriterion};

/// The training data for the ensemble.
///
/// The outer vector is the ensemble members.
/// The middle vector is the meta-ML models.
/// The inner vector is the epoch training data.
/// The tuple is the anomaly ratios and roc scores.
pub type TrainingData = Vec<Vec<(Vec<Vec<f32>>, Vec<f32>)>>;

/// A CHAODA ensemble.
#[derive(Serialize, Deserialize)]
pub struct Chaoda {
    /// The combination of the CHAODA algorithms and the meta-ML models.
    algorithms: Vec<(Member, Vec<MlModel>)>,
    /// The minimum depth of `Cluster`s to consider for selection.
    min_depth: usize,
}

impl Default for Chaoda {
    fn default() -> Self {
        Self {
            algorithms: Member::default_members()
                .into_par_iter()
                .map(|member| (member, MlModel::defaults()))
                .collect(),
            min_depth: 4,
        }
    }
}

impl Chaoda {
    /// Create a new `Chaoda` ensemble.
    #[must_use]
    pub const fn new(algorithms: Vec<(Member, Vec<MlModel>)>, min_depth: usize) -> Self {
        Self { algorithms, min_depth }
    }

    /// Get the number of predictors in the ensemble.
    #[must_use]
    pub fn num_predictors(&self) -> usize {
        self.algorithms.iter().map(|(_, models)| models.len()).sum()
    }

    /// Save the model to a given path.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to save the model to.
    ///
    /// # Errors
    ///
    /// * If there is an error creating the file.
    /// * If there is an error serializing the model.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("Error creating file: {e}"))?;
        bincode::serialize_into(file, self).map_err(|e| format!("Error serializing: {e}"))?;
        Ok(())
    }

    /// Load the model from a given path.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to load the model from.
    ///
    /// # Errors
    ///
    /// * If there is an error opening the file.
    /// * If there is an error deserializing the model.
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("Error opening file: {e}"))?;
        let model = bincode::deserialize_from(file).map_err(|e| format!("Error deserializing: {e}"))?;
        Ok(model)
    }

    /// Predict the anomaly scores for the given dataset and root `Cluster`.
    ///
    /// This method produces scores for points in their original order.
    ///
    /// # Arguments
    ///
    /// * `data`: The dataset to predict on.
    /// * `num_trees`: The number of trees to use in the ensemble.
    /// * `criteria`: The partition criterion to use for building the trees.
    /// * `seed`: The seed to use for random number generation, if any.
    ///
    /// # Returns
    ///
    /// The anomaly scores for each point in the dataset.
    pub fn predict<I, U, D, C>(&self, data: &D, root: &C) -> Vec<f32>
    where
        I: Instance,
        U: Number,
        D: Dataset<I, U> + Clone,
        C: OddBall<U>,
    {
        let permutation = data
            .permuted_indices()
            .map_or_else(|| (0..data.cardinality()).collect(), <[usize]>::to_vec);

        let mut graphs = self.create_graphs(data, root);
        let predictions = self
            .algorithms
            .par_iter()
            .zip(graphs.par_iter_mut())
            .flat_map(|((member, _), m_graphs)| {
                m_graphs
                    .par_iter_mut()
                    .map(|g| {
                        let scores = member.evaluate_points(g);
                        let mut scores = scores.into_iter().zip(permutation.iter()).collect::<Vec<_>>();
                        scores.sort_by_key(|(_, &i)| i);
                        scores.into_iter().map(|(s, _)| s).collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::aggregate_predictions(&predictions)
    }

    /// Aggregate the predictions of the ensemble.
    ///
    /// For now, we take the mean of the anomaly scores for each point. Later,
    /// we may want to consider other aggregation methods.
    #[must_use]
    pub fn aggregate_predictions(scores: &[Vec<f32>]) -> Vec<f32> {
        // Take the mean of the anomaly scores for each point
        let shape = (scores.len(), scores[0].len());
        let scores = scores.iter().flat_map(Vec::as_slice).copied().collect::<Vec<_>>();
        Array2::from_shape_vec(shape, scores)
            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."))
            .mean_axis(Axis(0))
            .unwrap_or_else(|| unreachable!("We made sure the shape was correct."))
            .to_vec()
    }

    /// Train the ensemble on the given datasets.
    ///
    /// # Arguments
    ///
    /// * `datasets`: The datasets and labels to train on.
    /// * `num_epochs`: The number of epochs to train for.
    /// * `criteria`: The partition criterion to use for building the tree.
    /// * `previous_data`: The previous training data to start from, if any.
    /// * `seed`: The seed to use for random number generation, if any.
    ///
    /// # Type Parameters
    ///
    /// * `I`: The type of the instances in the dataset.
    /// * `U`: The type of the distance values.
    /// * `D`: The type of the dataset.
    /// * `C`: The type of the `OddBall` `Cluster`.
    /// * `N`: Half the number of anomaly properties in the `Cluster`.
    /// * `P`: The partition criteria.
    pub fn train<I, U, D, C, P>(
        &mut self,
        datasets: &[(D, Vec<bool>)],
        num_epochs: usize,
        criteria: &P,
        previous_data: Option<TrainingData>,
        seed: Option<u64>,
    ) where
        I: Instance,
        U: Number,
        D: Dataset<I, U> + Clone,
        C: OddBall<U>,
        P: PartitionCriterion<U>,
    {
        let mut fresh_start = previous_data.is_none();

        let mut full_training_data = previous_data.unwrap_or_else(|| {
            self.algorithms
                .iter()
                .map(|(_, models)| models.iter().map(|_| (Vec::new(), Vec::new())).collect::<Vec<_>>())
                .collect()
        });
        let mut graphs;

        for e in 0..num_epochs {
            let training_data_size = full_training_data
                .iter()
                .map(|m| m.iter().map(|m| m.0.len()).sum::<usize>())
                .sum::<usize>();
            println!(
                "Starting Epoch {}/{num_epochs} with dataset size: {training_data_size}",
                e + 1
            );

            for (data, labels) in datasets {
                // Build the tree
                let mut data = data.clone();
                let seed = seed.map(|s| s + (e + data.cardinality()) as u64);
                let root = C::new_root(&data, seed).partition(&mut data, criteria, seed);

                // Create the graphs
                graphs = if fresh_start {
                    fresh_start = false;
                    let cluster_scorer = |clusters: &[&C]| {
                        clusters
                            .iter()
                            .map(|c| {
                                if c.depth() == self.min_depth || (c.is_leaf() && c.depth() < self.min_depth) {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    let graph = Graph::from_tree(&root, &data, cluster_scorer, 4);
                    self.algorithms
                        .iter()
                        .map(|(_, models)| models.iter().map(|_| graph.clone()).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                } else {
                    self.create_graphs(&data, &root)
                };

                // reorder labels to match permutation in data
                let permutation = data
                    .permuted_indices()
                    .map_or_else(|| (0..data.cardinality()).collect::<Vec<_>>(), <[usize]>::to_vec);
                let y_true = permutation
                    .iter()
                    .map(|&i| if labels[i] { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>();
                // Create the new training data
                let new_training_data = self.generate_training_data(&mut graphs, &y_true);

                // Aggregate the training data
                full_training_data = full_training_data
                    .into_iter()
                    .zip(new_training_data)
                    .map(|(m_old, m_new)| {
                        m_old
                            .into_iter()
                            .zip(m_new)
                            .map(|((mut x_old, mut y_old), (mut x_new, mut y_new))| {
                                x_old.append(&mut x_new);
                                y_old.append(&mut y_new);
                                (x_old, y_old)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                // Train the inner models
                self.train_inner_models(&full_training_data);

                // Report the ROC score
                let y_true = labels.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();
                let roc_score = {
                    let predictions = self.predict::<_, _, _, C>(&data, &root);
                    roc_auc_score(&y_true, &predictions)
                };
                println!(
                    "Epoch: {}/{num_epochs} ROC Score: {roc_score:.6} on Dataset: {}",
                    e + 1,
                    data.name()
                );
            }
        }
    }

    /// Create `Graph`s for the ensemble.
    fn create_graphs<I, U, D, C>(&self, data: &D, root: &C) -> Vec<Vec<Graph<U>>>
    where
        I: Instance,
        U: Number,
        D: Dataset<I, U>,
        C: OddBall<U>,
    {
        self.algorithms
            .par_iter()
            .map(|(_, models)| {
                models
                    .par_iter()
                    .map(|ml_model| {
                        let cluster_scorer = |clusters: &[&C]| {
                            let properties = clusters.iter().map(|c| c.ratios()).collect::<Vec<_>>();
                            ml_model
                                .predict(&properties)
                                .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."))
                        };
                        Graph::from_tree(root, data, cluster_scorer, self.min_depth)
                    })
                    .collect()
            })
            .collect()
    }

    /// Generate training data from `Graph`s.
    fn generate_training_data<U: Number>(&self, graphs: &mut [Vec<Graph<U>>], labels: &[f32]) -> TrainingData {
        self.algorithms
            .par_iter()
            .zip(graphs.par_iter_mut())
            .map(|((member, _), m_graphs)| {
                m_graphs
                    .par_iter_mut()
                    .map(|g| {
                        let train_x = g.iter_anomaly_properties().cloned().collect::<Vec<_>>();
                        let anomaly_ratings = Member::normalize_scores(&member.evaluate_clusters(g));
                        let train_y = g
                            .iter_clusters()
                            .zip(anomaly_ratings)
                            .map(|(&(start, cardinality), rating)| {
                                // The roc-score function needs both classes represented so we add a
                                // couple of dummy values to the end of the vectors.
                                let mut y_true = labels[start..(start + cardinality)].to_vec();
                                y_true.push(1.0);
                                y_true.push(0.0);
                                let mut y_pred = vec![rating; cardinality];
                                y_pred.push(1.0);
                                y_pred.push(0.0);
                                roc_auc_score(&y_true, &y_pred).as_f32()
                            })
                            .collect();
                        (train_x, train_y)
                    })
                    .collect()
            })
            .collect()

        // // Apply cross-pollination, i.e. run every member algo on all graphs that
        // // were built for OTHER member algorithms.
        // // This helps multiply the amount of training data.
        // self.algorithms
        //     .iter()
        //     .enumerate()
        //     .zip(training_data.iter_mut())
        //     .for_each(|((i, (member, _)), member_data)| {
        //         graphs
        //             .par_iter_mut()
        //             .enumerate()
        //             .zip(member_data.par_iter_mut())
        //             .for_each(|((j, m_graphs), (train_x, train_y))| {
        //                 if i != j {
        //                     for g in m_graphs.iter_mut() {
        //                         train_x.extend(g.iter_anomaly_properties().cloned());
        //                         let anomaly_ratings = Member::normalize_scores(&member.evaluate_clusters(g));
        //                         train_y.extend(g.iter_clusters().zip(anomaly_ratings).map(
        //                             |(&(start, cardinality), rating)| {
        //                                 // The roc-score function needs both classes represented so we add a
        //                                 // couple of dummy values to the end of the vectors.
        //                                 let mut y_true = labels[start..(start + cardinality)].to_vec();
        //                                 y_true.push(1.0);
        //                                 y_true.push(0.0);
        //                                 let mut y_pred = vec![rating; cardinality];
        //                                 y_pred.push(1.0);
        //                                 y_pred.push(0.0);
        //                                 roc_auc_score(&y_true, &y_pred).as_f32()
        //                             },
        //                         ));
        //                     }
        //                 }
        //             });
        //     });

        // training_data
    }

    /// Train the inner models given the training data.
    fn train_inner_models(&mut self, training_data: &TrainingData) {
        self.algorithms
            .par_iter_mut()
            .zip(training_data)
            .for_each(|((_, ml_models), member_data)| {
                ml_models
                    .par_iter_mut()
                    .zip(member_data)
                    .for_each(|(model, (train_x, train_y))| {
                        model.train(train_x, train_y).unwrap_or_else(|e| unreachable!("{e}"));
                    });
            });
    }

    /// Compute the ROC AUC score.
    #[allow(clippy::ptr_arg)]
    #[must_use]
    pub fn roc_auc_score(y_true: &Vec<f32>, y_pred: &Vec<f32>) -> f32 {
        roc_auc_score(y_true, y_pred).as_f32()
    }
}
