use std::sync::Arc;

use dashmap::{DashMap, DashSet};

pub use crate::cluster::Cluster;
pub use crate::dataset::Dataset;
pub use crate::graph::Edge;
pub use crate::graph::Graph;
pub use crate::manifold::Manifold;
pub use crate::metric::{metric_new, Metric, Number};

pub type Index = usize;
pub type Indices = Vec<Index>;
pub type Candidates<T, U> = DashMap<Arc<Cluster<T, U>>, U>;
pub type CandidatesMap<T, U> = DashMap<Arc<Cluster<T, U>>, Candidates<T, U>>;
pub type EdgesDict<T, U> = DashMap<Arc<Cluster<T, U>>, Arc<DashSet<Arc<Edge<T, U>>>>>;
