use crate::h5data;
use crate::h5number;

#[derive(Debug, Clone)]
pub struct H5Space<'a, H, T>
where
    H: h5number::H5Number,
    T: clam::Number,
{
    data: &'a h5data::H5Data<H>,
    metric: &'a dyn clam::Metric<T>,
}

impl<'a, H, T> H5Space<'a, H, T>
where
    H: h5number::H5Number,
    T: clam::Number,
{
    #[allow(dead_code)]
    pub fn new(data: &'a h5data::H5Data<H>, metric: &'a dyn clam::Metric<T>) -> Self {
        Self { data, metric }
    }
}

impl<'a, H, T> clam::Space<'a, T> for H5Space<'a, H, T>
where
    H: h5number::H5Number,
    T: clam::Number,
{
    fn data(&self) -> &dyn clam::Dataset<'a, T> {
        self.data
    }

    fn metric(&self) -> &dyn clam::Metric<T> {
        self.metric
    }
}
