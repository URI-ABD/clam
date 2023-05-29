use super::cluster::Cluster;
use super::dataset::Dataset;
use super::number::Number;

#[allow(dead_code)]
pub struct Manifold<'a, T: Number, U: Number, D: Dataset<T, U>> {
    root: Cluster<'a, T, U, D>,
}

// pub type CandidateNeighbors<T, U> = Vec<(Box<Cluster<T, U>>, U)>;
// candidate_neighbors: Option<CandidateNeighbors<T, U>>,

impl<'a, T: Number, U: Number, D: Dataset<T, U>> Manifold<'a, T, U, D> {}
