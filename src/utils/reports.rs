#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
pub struct ClusterReport {
    pub name: String,
    pub cardinality: usize,
    pub indices: Option<Vec<usize>>,
    pub arg_center: Option<usize>,
    pub arg_radius: Option<usize>,
    pub radius: Option<f64>,
    pub lfd: Option<f64>,
    pub ratios: Option<[f64; 6]>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
pub struct TreeReport {
    pub data_name: String,
    pub cardinality: usize,
    pub dimensionality: usize,
    pub metric_name: String,
    pub root_name: String,
    pub max_depth: usize,
    pub build_time: f64,
}

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone, Default)]
pub struct SearchReport {
    pub history: Vec<(String, f64)>,
    pub leaves: Vec<String>,
    pub hits: Vec<usize>,
    pub distances: Vec<Option<f64>>,
    pub num_distance_calls: usize,
}
