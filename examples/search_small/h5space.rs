use crate::h5data;
use crate::h5number;

#[derive(Debug, Clone)]
pub struct H5Space<'a, Tr: h5number::H5Number, T: clam::Number> {
    data: &'a h5data::H5Data<Tr>,
    metric: &'a dyn clam::Metric<T>,
}

impl<'a, Tr: h5number::H5Number, T: clam::Number> H5Space<'a, Tr, T> {
    #[allow(dead_code)]
    pub fn new(data: &'a h5data::H5Data<Tr>, metric: &'a dyn clam::Metric<T>) -> Self {
        Self { data, metric }
    }
}

impl<'a, Tr: h5number::H5Number, T: clam::Number> clam::Space<'a, T> for H5Space<'a, Tr, T> {
    fn data(&self) -> &dyn clam::Dataset<'a, T> {
        self.data
    }

    fn metric(&self) -> &dyn clam::Metric<T> {
        self.metric
    }
}
