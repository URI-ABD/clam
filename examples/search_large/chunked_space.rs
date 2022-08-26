use super::chunked_data::ChunkedTabular;

use clam::prelude::*;

pub struct ChunkedTabularSpace<'a, T: Number, U: Number> {
    data: &'a ChunkedTabular<'a, T>,
    metric: &'a dyn clam::Metric<T, U>,
    uses_cache: bool,
    cache: clam::Cache<U>,
}

impl<'a, T: clam::Number, U: clam::Number> ChunkedTabularSpace<'a, T, U> {
    pub fn new(data: &'a ChunkedTabular<T>, metric: &'a dyn clam::Metric<T, U>, use_cache: bool) -> Self {
        ChunkedTabularSpace {
            data,
            metric,
            uses_cache: use_cache,
            cache: std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }
}

impl<'a, T: clam::Number, U: clam::Number> std::fmt::Debug for ChunkedTabularSpace<'a, T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space")
            .field("data", &self.data.name())
            .field("metric", &self.metric.name())
            .field("uses_cache", &self.uses_cache)
            .finish()
    }
}

impl<'a, T: Number, U: Number> Space<'a, T, U> for ChunkedTabularSpace<'a, T, U> {
    fn data(&self) -> &dyn Dataset<'a, T> {
        self.data
    }

    fn metric(&self) -> &dyn Metric<T, U> {
        self.metric
    }

    fn uses_cache(&self) -> bool {
        self.uses_cache
    }

    fn cache(&self) -> clam::Cache<U> {
        self.cache.clone()
    }
}
