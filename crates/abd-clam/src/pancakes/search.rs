//! Supplies the `Search` trait for performing RNN- and KNN-Search in a compressed space.

use distances::number::UInt;

use crate::Instance;

use super::{knn, rnn, CodecData};

impl<I: Instance, U: UInt, M: Instance> CodecData<I, U, M> {
    /// Performs an RNN-Search.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `radius` - The radius to use for the search.
    /// * `algo` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn rnn_search(&self, query: &I, radius: U, algo: &rnn::Algorithm) -> Vec<(usize, U)> {
        algo.search(query, radius, self)
    }

    /// Performs a KNN-Search.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `k` - The number of neighbors to search for.
    /// * `algo` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn knn_search(&self, query: &I, k: usize, algo: &knn::Algorithm) -> Vec<(usize, U)> {
        algo.search(query, k, self)
    }
}

#[cfg(test)]
mod tests {
    use distances::strings::levenshtein;

    use super::*;

    use crate::{
        pancakes::{decode_general, encode_general, CodecData, SquishyBall},
        Cluster, PartitionCriteria, VecDataset,
    };

    fn lev_metric(x: &String, y: &String) -> u16 {
        levenshtein(x, y)
    }

    #[test]
    fn test_rnn_search() -> Result<(), String> {
        let strings = vec![
            "NAJIBPEPPERS-EATS".to_string(),
            "NAJIB-PEPPERSEATS".to_string(),
            "NAJIB-EATSPEPPERS".to_string(),
            "NAJIBEATS-PEPPERS".to_string(),
            "TOM-EATSWHATFOODEATS".to_string(),
            "TOMEATSWHATFOOD-EATS".to_string(),
            "FOODEATS-WHATTOMEATS".to_string(),
            "FOODEATSWHAT-TOMEATS".to_string(),
        ];

        let mut dataset = VecDataset::new("test-codec".to_string(), strings, lev_metric, true);
        let criteria = PartitionCriteria::default();
        let seed = Some(42);
        let root = SquishyBall::new_root(&dataset, seed).partition(&mut dataset, &criteria, seed);

        let metadata = dataset.metadata().to_vec();
        let dataset = CodecData::new(root, &dataset, encode_general::<u16>, decode_general, metadata)?;

        let query = "NAJIBEATSPEPPERS".to_string();
        let radius = 2;

        for algo in [rnn::Algorithm::Linear, rnn::Algorithm::Clustered] {
            let result = dataset.rnn_search(&query, radius, &algo);

            println!("{}: {result:?}", algo.name());
            assert_eq!(result.len(), 2);
        }

        Ok(())
    }

    #[test]
    fn test_knn_search() -> Result<(), String> {
        let strings = vec![
            "NAJIBPEPPERS-EATS".to_string(),
            "NAJIB-PEPPERSEATS".to_string(),
            "NAJIB-EATSPEPPERS".to_string(),
            "NAJIBEATS-PEPPERS".to_string(),
            "TOM-EATSWHATFOODEATS".to_string(),
            "TOMEATSWHATFOOD-EATS".to_string(),
            "FOODEATS-WHATTOMEATS".to_string(),
            "FOODEATSWHAT-TOMEATS".to_string(),
        ];

        let mut dataset = VecDataset::new("test-codec".to_string(), strings, lev_metric, true);
        let criteria = PartitionCriteria::default();
        let seed = Some(42);
        let root = SquishyBall::new_root(&dataset, seed).partition(&mut dataset, &criteria, seed);

        let metadata = dataset.metadata().to_vec();
        let codec_dataset = CodecData::new(root, &dataset, encode_general::<u16>, decode_general, metadata)?;

        let query = "NAJIBEATSPEPPERS".to_string();
        let k = 2;

        for algo in [knn::Algorithm::Linear] {
            let result = codec_dataset.knn_search(&query, k, &algo);

            println!("{}: {result:?}", algo.name());
            assert_eq!(
                [dataset[result[0].0].clone(), dataset[result[1].0].clone()],
                ["NAJIBEATS-PEPPERS", "NAJIB-EATSPEPPERS"]
            );
        }

        Ok(())
    }
}
