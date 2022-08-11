use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use crate::h5data;
use crate::h5number;

#[derive(Debug, Clone)]
pub struct H5Space<'a, Tr: h5number::H5Number, T: clam::Number, U: clam::Number> {
    data: &'a h5data::H5Data<Tr>,
    metric: &'a dyn clam::Metric<T, U>,
    uses_cache: bool,
    cache: clam::Cache<U>,
}

impl<'a, Tr: h5number::H5Number, T: clam::Number, U: clam::Number> H5Space<'a, Tr, T, U> {
    #[allow(dead_code)]
    pub fn new(data: &'a h5data::H5Data<Tr>, metric: &'a dyn clam::Metric<T, U>, use_cache: bool) -> Self {
        Self {
            data,
            metric,
            uses_cache: use_cache,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl<'a, Tr: h5number::H5Number, T: clam::Number, U: clam::Number> clam::Space<T, U> for H5Space<'a, Tr, T, U> {
    fn data(&self) -> &dyn clam::Dataset<T> {
        self.data
    }

    fn metric(&self) -> &dyn clam::Metric<T, U> {
        self.metric
    }

    fn uses_cache(&self) -> bool {
        self.uses_cache
    }

    fn cache(&self) -> clam::Cache<U> {
        self.cache.clone()
    }
}
