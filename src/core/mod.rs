mod cluster;
mod graph;
mod manifold;

pub mod criteria;

pub use cluster::Cluster;
pub use graph::Edge;
pub use graph::Graph;
pub use manifold::Manifold;

pub use cluster::ClusterName;

// TODO: Break out a struct for a cluster-cache that will store intra-cluster information
//       that does not make sense to store as a member in cluster/graph/manifold
//       e.g. cluster -> candidates relationships
//            cluster -> parent-child ratios
