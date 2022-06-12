#[derive(Debug, Clone)]
pub struct H5Space<'a, T: crate::h5number::H5Number, U: clam::Number> {
    data: &'a crate::h5data::H5Data,
    metric: &'a dyn clam::Metric<T, U>,
    uses_cache: bool,
    cache: clam::traits::space::Cache<U>,
}

impl<'a, T: crate::h5number::H5Number, U: clam::Number> H5Space<'a, T, U> {
    #[allow(dead_code)]
    pub fn new(data: &'a crate::h5data::H5Data, metric: &'a dyn clam::Metric<T, U>, use_cache: bool) -> Self {
        Self {
            data,
            metric,
            uses_cache: use_cache,
            cache: std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }

    #[allow(dead_code)]
    pub fn as_space(&self) -> &dyn clam::Space<T, U> {
        self
    }
}

impl<'a, T: crate::h5number::H5Number, U: clam::Number> clam::Space<T, U> for H5Space<'a, T, U> {
    fn data(&self) -> &dyn clam::Dataset<T> {
        self.data
    }

    fn metric(&self) -> &dyn clam::Metric<T, U> {
        self.metric
    }

    fn uses_cache(&self) -> bool {
        self.uses_cache
    }

    fn cache(&self) -> clam::traits::space::Cache<U> {
        self.cache.clone()
    }
}
