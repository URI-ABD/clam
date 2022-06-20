use std::sync::Arc;

use crate::{get_individual_algorithms, prelude::*};

pub struct MetaML<T: Number, U: Number> {
    pub mml_name: String,
    pub metric: String,
    pub algorithm_name: String,
    pub mml_method: criteria::MetaMLScorer,
    pub algorithm: Arc<crate::anomaly::IndividualAlgorithm<T, U>>,
}

impl<T: Number, U: Number> std::fmt::Display for MetaML<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}_{}", self.mml_name, self.metric, self.algorithm_name)
    }
}

pub fn get_meta_ml_methods<T: Number, U: Number>() -> Vec<MetaML<T, U>> {
    let algorithms = get_individual_algorithms();

    crate::anomaly::get_meta_ml_scorers()
        .into_iter()
        .map(|(name, mml_method)| {
            let name = name.split('_').collect::<Vec<_>>();
            let (mml_name, metric, algorithm_name) = (name[0].to_string(), name[1].to_string(), name[2].to_string());
            let algorithm = algorithms
                .iter()
                .filter(|(name, _)| algorithm_name == *name)
                .map(|(_, algorithm)| algorithm)
                .next()
                .unwrap();

            MetaML {
                mml_name,
                metric,
                algorithm_name,
                mml_method,
                algorithm: Arc::clone(algorithm),
            }
        })
        .collect()
}
