use crate::prelude::*;

#[allow(dead_code)]
pub struct Manifold<'a, T: Number, U: Number> {
    root: Cluster<'a, T, U>,
}

// pub type CandidateNeighbors<T, U> = Vec<(Box<Cluster<T, U>>, U)>;
// candidate_neighbors: Option<CandidateNeighbors<T, U>>,

impl<'a, T: Number, U: Number> Manifold<'a, T, U> {
    #[allow(unused_mut)]
    pub fn with_candidate_neighbors(mut self) -> Self {
        // if self.depth() > 0 {
        //     panic!("Cluster Candidate Neighbors may only be set from the root cluster.")
        // }

        // self.candidate_neighbors = Some(vec![(Box::new(self), U::zero())]);
        self

        // todo!()
        // match self.candidate_neighbors.read().unwrap().clone() {
        //     Some(candidate_neighbors) => candidate_neighbors,
        //     None => match &self.parent {
        //         None => {
        //             let mut candidate_neighbors: CandidateNeighbors<T, U> = HashMap::new();
        //             candidate_neighbors.insert(self.clone(), U::zero());
        //             candidate_neighbors
        //         }
        //         Some(parent) => {
        //             let parent = Weak::upgrade(parent).unwrap();
        //             let mut candidates: CandidateNeighbors<T, U> = parent.clone().candidate_neighbors();
        //             let radius = if self.is_singleton() {
        //                 *candidates.get(&parent).unwrap()
        //             } else {
        //                 self.radius()
        //             };

        //             let non_leaf_candidates: Vec<_> = candidates.keys().filter(|&c| !c.is_leaf()).cloned().collect();
        //             if !non_leaf_candidates.is_empty() {
        //                 let children: Vec<ACluster<T, U>> = non_leaf_candidates
        //                     .into_iter()
        //                     .flat_map(|c| {
        //                         let c = c.children().unwrap();
        //                         [c.0, c.1]
        //                     })
        //                     .collect();
        //                 let arg_centers: Vec<_> = children.iter().map(|c| c.arg_center()).collect();
        //                 let distances = self
        //                     .space
        //                     .distance_one_to_many(self.arg_center(), &arg_centers)
        //                     .to_vec();
        //                 candidates.extend(
        //                     children
        //                         .iter()
        //                         .zip(distances.iter())
        //                         .filter(|(c, &d)| d <= (c.radius() + radius * U::from(4u64).unwrap()))
        //                         .map(|(c, &d)| (Arc::clone(c), d)),
        //                 );
        //                 candidates.insert(self.clone(), self.radius());
        //                 *self.candidate_neighbors.write().unwrap() = Some(candidates.clone());
        //             }
        //             candidates
        //         }
        //     },
        // }
    }

    // pub fn candidate_neighbors(&self) -> &[(Box<Cluster<T, U>>, U)] {
    //     self.candidate_neighbors
    //         .as_ref()
    //         .expect("Please call `with_candidate_neighbors` on the root before using this method.")
    // }
}
