//! Anomaly detection algorithms using CLAM.
//!
//! This module contains CLAM-CHAODA (Clustered Hierarchical Anomaly and Outlier Detection Algorithms). This is a family of algorithms that use CLAM trees to
//! impute graphs that enable unsupervised anomaly detection algorithms.
//!
//! For trees, this enables the [`Tree::annotate_anomaly_features`](crate::Tree::annotate_anomaly_features) method (along with its parallel version). These
//! features can then be used for creating CHAODA graphs. The graphs can, in turn, be used for anomaly detection using the algorithms we provide...

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    chaoda::{algorithms::GraphAlgorithm, meta_ml::MetaMlModel},
};

mod features;
mod graph;
pub mod meta_ml;
mod training;
mod tree;

pub mod algorithms;

pub use features::{AnomalyFeatures, normalize_features};
pub use graph::{Component, Graph, Node};
pub use training::AnomalyLabel;

/// A graph algorithm and its associated meta-ml algorithms.
pub(crate) type GraphEnsemble = (GraphAlgorithm, Vec<MetaMlModel>);

/// Unsupervised anomaly detection with CHAODA.
pub struct Chaoda {
    /// The suite of graph algorithms and associated meta-ml models to use for anomaly detection.
    model_suite: Vec<GraphEnsemble>,
}

impl Chaoda {
    /// Train a Chaoda ensemble with the given tree.
    ///
    /// # Errors
    ///
    /// - Any roc-scores fail to be computed.
    /// - Any of the meta-ml models fails to train.
    pub fn train<Id, I, T, A, M>(tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Self, String>
    where
        Id: AnomalyLabel,
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        let algs: Vec<GraphAlgorithm> = vec![
            algorithms::AccumulatedCardinalityRatios.into(),
            algorithms::GraphNeighborhoodSize.into(),
            algorithms::RelativeClusterCardinality.into(),
            algorithms::RelativeComponentCardinality.into(),
            algorithms::RelativeVertexDegree.into(),
            algorithms::StationaryProbabilities.into(),
        ];

        let mut data = training::MetaMlTrainingData::new(algs);
        data.extract_data(tree)?;
        let model_suite = data.train_models()?;

        Ok(Self { model_suite })
    }

    /// Predict anomaly rankings for the items in the given tree.
    ///
    /// After computing all rankings, use the [`fuse_rankings`] function to combine the rankings into a single ranking.
    pub fn predict<Id, I, T, A, M>(&self, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<Vec<usize>>
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        let mut rankings = Vec::new();

        for (alg, models) in &self.model_suite {
            for model in models {
                let (directly_selected, ancestors) = tree.select_chaoda_clusters(model, 4);
                let graph = Graph::from_tree(tree, &directly_selected, &ancestors);
                rankings.push(alg.rank_items(&graph, tree));
            }
        }

        rankings
    }

    /// Parallel version of [`train`](Self::train).
    ///
    /// # Errors
    ///
    /// - See [`train`](Self::train) for error conditions.
    pub fn par_train<Id, I, T, A, M>(tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Self, String>
    where
        Id: AnomalyLabel + Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        let algs = vec![
            algorithms::AccumulatedCardinalityRatios.into(),
            algorithms::GraphNeighborhoodSize.into(),
            algorithms::RelativeClusterCardinality.into(),
            algorithms::RelativeComponentCardinality.into(),
            algorithms::RelativeVertexDegree.into(),
            algorithms::StationaryProbabilities.into(),
        ];

        let mut data = training::MetaMlTrainingData::new(algs);
        data.par_extract_data(tree)?;
        let model_suite = data.par_train_models()?;

        Ok(Self { model_suite })
    }

    /// Parallel version of [`predict`](Self::predict).
    pub fn par_predict<Id, I, T, A, M>(&self, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<Vec<usize>>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        let mut rankings = Vec::new();

        for (alg, models) in &self.model_suite {
            rankings.extend(
                models
                    .par_iter()
                    .map(|model| {
                        let (directly_selected, ancestors) = tree.par_select_chaoda_clusters(model, 4);
                        let graph = Graph::par_from_tree(tree, &directly_selected, &ancestors);
                        alg.rank_items(&graph, tree)
                    })
                    .collect::<Vec<_>>(),
            );
        }

        rankings
    }
}

/// Apply reciprocal rank fusion to the rankings produced by the CHAODA ensemble.
///
/// DOI: 10.1145/1571941.1572114
#[must_use]
pub fn fuse_rankings(rankings: &[Vec<usize>]) -> Vec<f64> {
    let mut fused_rankings = vec![0_f64; rankings[0].len()];
    for ranks in rankings {
        #[expect(clippy::cast_precision_loss)]
        for (f, r) in fused_rankings.iter_mut().zip(ranks) {
            *f += ((r + 60) as f64).recip();
        }
    }
    fused_rankings
}

/// Compute the ROC AUC score for binary classification.
///
/// # Arguments
///
/// * `y_true` - An iterator of boolean values representing the true binary labels.
/// * `y_pred` - A vector of floating-point values representing the predicted probabilities for the positive class.
///
/// # Returns
///
/// The ROC AUC score as a floating-point value between 0.0 and 1.0.
///
/// # Errors
///
/// - If the lengths of `y_true` and `y_pred` do not match.
pub fn roc_auc_score(y_true: &[bool], y_pred: &Vec<f64>) -> Result<f64, String> {
    let y_true = y_true.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    if y_true.len() == y_pred.len() {
        Ok(smartcore::metrics::roc_auc_score(&y_true, y_pred))
    } else {
        Err(format!(
            "Length mismatch: y_true has length {}, but y_pred has length {}",
            y_true.len(),
            y_pred.len()
        ))
    }
}
