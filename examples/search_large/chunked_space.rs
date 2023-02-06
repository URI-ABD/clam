use super::chunked_data::ChunkedTabular;

use clam::prelude::*;

pub struct ChunkedTabularSpace<'a, T: Number> {
    data: &'a ChunkedTabular<'a, T>,
    metric: &'a dyn clam::Metric<T>,
}

impl<'a, T: clam::Number> ChunkedTabularSpace<'a, T> {
    pub fn new(data: &'a ChunkedTabular<T>, metric: &'a dyn clam::Metric<T>) -> Self {
        ChunkedTabularSpace { data, metric }
    }
}

impl<'a, T: clam::Number> std::fmt::Debug for ChunkedTabularSpace<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space")
            .field("data", &self.data.name())
            .field("metric", &self.metric.name())
            .finish()
    }
}

impl<'a, T: Number> Space<'a, T> for ChunkedTabularSpace<'a, T> {
    fn data(&self) -> &dyn Dataset<'a, T> {
        self.data
    }

    fn metric(&self) -> &dyn Metric<T> {
        self.metric
    }
}
