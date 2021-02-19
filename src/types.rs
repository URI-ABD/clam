use std::sync::Arc;

use dashmap::{DashMap, DashSet};

use crate::cluster::Cluster;
use crate::graph::Edge;

pub type Index = usize;
pub type Indices = Vec<Index>;
pub type Candidates<T, U> = DashMap<Arc<Cluster<T, U>>, U>;
pub type CandidatesMap<T, U> = DashMap<Arc<Cluster<T, U>>, Candidates<T, U>>;
pub type EdgesDict<T, U> = DashMap<Arc<Cluster<T, U>>, Arc<DashSet<Arc<Edge<T, U>>>>>;
