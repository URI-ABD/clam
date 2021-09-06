//! Implements Compression and Decompression for `Datasets` and `Clusters`.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::dataset::RowMajor;
use crate::{prelude::*, Cakes};

/// A `Dataset` that also allows for compression and decompression.
/// Instances in a `CompressibleDataset` can be encoded in terms of each other to
/// produce compressed encodings represented as bytes.
/// Those bytes can also be used to decode an encoded instance by using the reference.
pub trait CompressibleDataset<T: Number, U: Number>: Dataset<T, U> {
    /// Encode one instance in terms of another.
    fn encode(
        &self,
        reference: Index,
        target: Index,
    ) -> Result<Vec<u8>, String> {
        self.metric()
            .encode(&self.instance(reference), &self.instance(target))
    }

    /// Decode an instance from the encoded bytes and the reference.
    fn decode(
        &self,
        reference: Index,
        encoding: &[u8],
    ) -> Result<Vec<T>, String> {
        self.metric().decode(&self.instance(reference), encoding)
    }
}

impl<T: Number, U: Number> CompressibleDataset<T, U> for RowMajor<T, U> {}

impl<T: 'static + Number, U: 'static + Number> RowMajor<T, U> {
    pub fn as_arc_compressible_dataset(
        self: Arc<Self>,
    ) -> Arc<dyn CompressibleDataset<T, U>> {
        self
    }
}

pub struct PackableCluster<U: Number> {
    pub name: ClusterName,
    pub cardinality: usize,
    pub center: Vec<u8>,
    pub radius: U,
    pub encodings: Vec<Vec<u8>>,
}

impl<U: Number> PackableCluster<U> {
    pub fn from_cluster<T: Number>(
        cluster: Arc<Cluster<T, U>>,
        dataset: Arc<dyn CompressibleDataset<T, U>>,
        reference: Index,
        direct: bool,
    ) -> Result<Self, String> {
        let encodings: Result<Vec<_>, String> = if direct {
            let encodings: Vec<_> = cluster
                .indices
                .par_iter()
                .filter(|&&i| i != cluster.argcenter)
                .map(|&i| dataset.encode(reference, i))
                .collect();
            encodings.into_iter().collect()
        } else {
            Ok(vec![])
        };

        Ok(PackableCluster {
            name: cluster.name.clone(),
            cardinality: cluster.cardinality,
            center: dataset.encode(reference, cluster.argcenter)?,
            radius: cluster.radius,
            encodings: encodings?,
        })
    }

    pub fn depth(&self) -> usize {
        self.name.len() - 1
    }

    pub fn is_singleton(&self) -> bool {
        self.radius == U::from(0).unwrap()
    }

    pub fn decode_center<T: Number>(
        &self,
        metric: &Arc<dyn Metric<T, U>>,
        reference: &[T],
    ) -> Result<Vec<T>, String> {
        metric.decode(reference, &self.center)
    }

    pub fn decode_instances<T: Number>(
        &self,
        metric: &Arc<dyn Metric<T, U>>,
        reference: &[T],
    ) -> Result<Vec<Vec<T>>, String> {
        let center = self.decode_center(metric, reference)?;
        let mut instances: Vec<_> = self
            .encodings
            .iter()
            .map(|encoding| metric.decode(&center, encoding))
            .collect();
        instances.push(Ok(center));
        instances.into_iter().collect()
    }
}

/// A Vec of Clusters that overlap with the query ball.
type ClusterHits<U> = Vec<Arc<PackableCluster<U>>>;

/// A HashMap of indices of all hits and their distances to the query.
type Hits<T, U> = Vec<(Vec<T>, U)>;

pub struct Codec<T: Number, U: Number> {
    pub dataset: Arc<dyn CompressibleDataset<T, U>>,
    pub root: Arc<PackableCluster<U>>,
    pub center: Vec<T>,
    pub tree_map: HashMap<ClusterName, Arc<PackableCluster<U>>>,
}

impl<T: Number, U: Number> Codec<T, U> {
    pub fn from_cakes(
        _dataset: &Arc<dyn CompressibleDataset<T, U>>,
        _cakes: &Cakes<T, U>,
    ) -> Self {
        todo!()
    }

    pub fn diameter(&self) -> U {
        U::from(2).unwrap() * self.root.radius
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        self.dataset.metric().distance(x, y)
    }

    pub fn rnn_instances(&self, query: &[T], radius: Option<U>) -> Vec<Vec<T>> {
        self.rnn(query, radius)
            .into_iter()
            .map(|(instance, _)| instance)
            .collect()
    }

    pub fn rnn(&self, query: &[T], radius: Option<U>) -> Hits<T, U> {
        self.leaf_search(query, radius, self.tree_search(query, radius))
    }

    pub fn tree_search(&self, query: &[T], radius: Option<U>) -> ClusterHits<U> {
        // parse the search radius
        let radius = radius.unwrap_or_else(U::zero);
        // if query ball has overlapping volume with the root, delegate to the recursive, private method.
        if self.distance(&self.center, query) <= (radius + self.root.radius) {
            self._tree_search(&self.root, query, radius)
        } else {
            // otherwise, return an empty Vec signifying no possible hits.
            vec![]
        }
    }

    fn _tree_search(
        &self,
        _cluster: &Arc<PackableCluster<U>>,
        _query: &[T],
        _radius: U,
    ) -> ClusterHits<U> {
        todo!()
    }

    pub fn leaf_search(
        &self,
        _query: &[T],
        _radius: Option<U>,
        _clusters: ClusterHits<U>,
    ) -> Hits<T, U> {
        todo!()
    }

    pub fn linear_search_instances(
        &self,
        query: &[T],
        radius: Option<U>,
        instances: Vec<Vec<T>>,
    ) -> Vec<Vec<T>> {
        self.linear_search(query, radius, instances)
            .into_iter()
            .map(|(instance, _)| instance)
            .collect()
    }

    pub fn linear_search(
        &self,
        query: &[T],
        radius: Option<U>,
        instances: Vec<Vec<T>>,
    ) -> Hits<T, U> {
        let radius = radius.unwrap_or_else(U::zero);
        instances
            .into_par_iter()
            .map(|instance| {
                let distance = self.distance(query, &instance);
                (instance, distance)
            })
            .filter(|(_, d)| *d <= radius)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dataset::RowMajor;
    use crate::metric_from_name;
    use crate::CompressibleDataset;

    #[test]
    fn test_codec() {
        let data = vec![
            vec![0., 0., 0.],
            vec![1., 1., 1.],
            vec![2., 2., 2.],
            vec![3., 3., 3.],
        ];
        let metric = metric_from_name("hamming").unwrap();
        let dataset: Arc<dyn CompressibleDataset<_, f64>> =
            Arc::new(RowMajor::new(Arc::new(data), metric, false));

        let encoded = dataset.encode(0, 1).unwrap();
        let decoded = dataset.decode(0, &encoded).unwrap();

        assert_eq!(dataset.instance(1), decoded);
    }
}
