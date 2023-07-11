use distances::Number;

use crate::{cluster::Tree, dataset::Dataset};

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
pub enum RnnAlgorithm {
    Linear,
    Clustered,
}

impl Default for RnnAlgorithm {
    fn default() -> Self {
        Self::Clustered
    }
}

impl RnnAlgorithm {
    pub fn search<T, U, D>(&self, query: T, radius: U, tree: &Tree<T, U, D>) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        match self {
            Self::Linear => Self::linear_search(tree.data(), query, radius, tree.indices()),
            Self::Clustered => Self::clustered_search(tree, query, radius),
        }
    }

    pub(crate) fn linear_search<T, U, D>(data: &D, query: T, radius: U, indices: &[usize]) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        let distances = data.query_to_many(query, indices);
        indices
            .iter()
            .copied()
            .zip(distances.into_iter())
            .filter(|&(_, d)| d <= radius)
            .collect()
    }

    pub(crate) fn clustered_search<T, U, D>(tree: &Tree<T, U, D>, query: T, radius: U) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        let data = tree.data();

        // Tree search.
        let [confirmed, straddlers] = {
            let mut confirmed = Vec::new();
            let mut straddlers = Vec::new();
            let mut candidates = vec![tree.root()];

            let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
            while !candidates.is_empty() {
                (terminal, non_terminal) = candidates
                    .into_iter()
                    .map(|c| (c, c.distance_to_instance(data, query)))
                    .filter(|&(c, d)| d <= (c.radius + radius))
                    .partition(|&(c, d)| (c.radius + d) <= radius);
                confirmed.append(&mut terminal);

                (terminal, non_terminal) = non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
                straddlers.append(&mut terminal);

                candidates = non_terminal
                    .into_iter()
                    .flat_map(|(c, d)| {
                        if d < c.radius {
                            c.overlapping_children(data, query, radius)
                        } else {
                            c.children().unwrap().to_vec()
                        }
                    })
                    .collect();
            }

            [confirmed, straddlers]
        };

        // Leaf Search
        let hits = confirmed.into_iter().flat_map(|(c, d)| {
            let distances = if c.is_singleton() {
                vec![d; c.cardinality]
            } else {
                data.query_to_many(query, c.indices(data))
            };
            c.indices(data).iter().copied().zip(distances.into_iter())
        });

        let indices = straddlers
            .into_iter()
            .flat_map(|(c, _)| c.indices(data))
            .copied()
            .collect::<Vec<_>>();

        hits.chain(Self::linear_search(data, query, radius, &indices).into_iter())
            .collect()
    }
}
