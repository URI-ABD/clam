use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Report<'a> {
    pub data_name: &'a str,
    pub metric_name: &'a str,
    pub num_queries: usize,
    pub num_runs: usize,
    pub cardinality: usize,
    pub dimensionality: usize,
    pub tree_depth: usize,
    pub build_time: f64,
    pub root_radius: f64,
    pub search_radii: Vec<f64>,
    pub search_times: Vec<Vec<f64>>,
    pub output_sizes: Vec<usize>,
    pub recalls: Vec<f64>,
}

impl<'a> Report<'a> {
    pub fn is_valid(&self) -> Vec<String> {
        let mut reasons = vec![];

        if self.num_queries != self.search_radii.len() {
            reasons.push(format!(
                "self.num_queries != self.search_radii.len(): {} != {}",
                self.num_queries,
                self.search_radii.len()
            ));
        }

        if self.num_queries != self.search_times.len() {
            reasons.push(format!(
                "self.num_queries != self.search_times.len(): {} != {}",
                self.num_queries,
                self.search_times.len()
            ));
        }

        if self.num_queries != self.output_sizes.len() {
            reasons.push(format!(
                "self.num_queries != self.output_sizes.len(): {} != {}",
                self.num_queries,
                self.output_sizes.len()
            ));
        }

        if self.num_queries != self.recalls.len() {
            reasons.push(format!(
                "self.num_queries != self.recalls.len(): {} != {}",
                self.num_queries,
                self.recalls.len()
            ));
        }

        for (i, samples) in self.search_times.iter().enumerate() {
            if self.num_runs != samples.len() {
                reasons.push(format!(
                    "{}/{}: self.num_runs != samples.len(): {} != {}",
                    i + 1,
                    self.num_queries,
                    self.num_runs,
                    samples.len()
                ));
            }
        }

        reasons
    }
}
