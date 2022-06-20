/// A `Box`ed function that assigns a score for a given `Cluster`.
pub type MetaMLScorer = Box<dyn (Fn(super::cluster::Ratios) -> f64) + Send + Sync>;
