use super::cluster::Tree;
use super::dataset::Dataset;
use super::number::Number;

#[allow(dead_code)]
pub struct Manifold<T: Number, U: Number, D: Dataset<T, U>> {
    root: Tree<T, U, D>,
}

// pub type CandidateNeighbors<T, U> = Vec<(Box<Cluster<T, U>>, U)>;
// candidate_neighbors: Option<CandidateNeighbors<T, U>>,

impl<T: Number, U: Number, D: Dataset<T, U>> Manifold<T, U, D> {}
