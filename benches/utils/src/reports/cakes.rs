//! Reports for Cakes results.

use distances::Number;

/// Reports for Cakes results search.
pub struct Results<T: Number> {
    /// The dataset name.
    dataset: String,
    /// The cardinality of the dataset.
    cardinality: usize,
    /// The dimensionality of the dataset.
    dimensionality: usize,
    /// The metric used.
    metric: String,
    /// A vector of:
    ///
    /// - The name of the `Cluster` type.
    /// - The name of the search algorithm.
    /// - The value of radius.
    /// - The mean time taken (seconds per query) to perform the search.
    /// - The mean throughput (queries per second).
    /// - The mean number of hits.
    /// - The mean recall.
    /// - The mean number of distance computations per query.
    #[allow(clippy::type_complexity)]
    radial_results: Vec<(String, String, T, f32, f32, f32, f32, f32)>,
    /// A vector of:
    ///
    /// - The name of the `Cluster` type.
    /// - The name of the search algorithm.
    /// - The value of k.
    /// - The mean time taken (seconds per query) to perform the search.
    /// - The mean throughput (queries per second).
    /// - The mean number of hits.
    /// - The mean recall.
    /// - The mean number of distance computations per query.
    #[allow(clippy::type_complexity)]
    k_results: Vec<(String, String, usize, f32, f32, f32, f32, f32)>,
}

impl<T: Number> Results<T> {
    /// Create a new report.
    #[must_use]
    pub fn new(data_name: &str, cardinality: usize, dimensionality: usize, metric: &str) -> Self {
        Self {
            dataset: data_name.to_string(),
            cardinality,
            dimensionality,
            metric: metric.to_string(),
            radial_results: Vec::new(),
            k_results: Vec::new(),
        }
    }

    /// Add a new result for radial search.
    #[allow(clippy::too_many_arguments)]
    pub fn append_radial_result(
        &mut self,
        cluster: &str,
        algorithm: &str,
        radius: T,
        time: f32,
        throughput: f32,
        output_sizes: &[usize],
        recalls: &[f32],
        distance_count: f32,
    ) {
        let mean_output_size = abd_clam::utils::mean(output_sizes);
        let mean_recall = abd_clam::utils::mean(recalls);
        self.radial_results.push((
            cluster.to_string(),
            algorithm.to_string(),
            radius,
            time,
            throughput,
            mean_output_size,
            mean_recall,
            distance_count,
        ));
        self.log_last_radial();
    }

    /// Add a new result for k-NN search.
    #[allow(clippy::too_many_arguments)]
    pub fn append_k_result(
        &mut self,
        cluster: &str,
        algorithm: &str,
        k: usize,
        time: f32,
        throughput: f32,
        output_sizes: &[usize],
        recalls: &[f32],
        distance_count: f32,
    ) {
        let mean_output_size = abd_clam::utils::mean(output_sizes);
        let mean_recall = abd_clam::utils::mean(recalls);
        self.k_results.push((
            cluster.to_string(),
            algorithm.to_string(),
            k,
            time,
            throughput,
            mean_output_size,
            mean_recall,
            distance_count,
        ));
        self.log_last_k();
    }

    /// Logs the last radial record.
    fn log_last_radial(&self) {
        let mut parts = vec![
            format!("Dataset: {}", self.dataset),
            format!("Cardinality: {}", self.cardinality),
            format!("Dimensionality: {}", self.dimensionality),
            format!("Metric: {}", self.metric),
        ];
        if let Some((cluster, algorithm, radius, time, throughput, output_size, recall, distance_computations)) =
            self.radial_results.last()
        {
            parts.push(format!("Cluster: {cluster}"));
            parts.push(format!("Algorithm: {algorithm}"));
            parts.push(format!("Radius: {radius}"));
            parts.push(format!("Time: {time}"));
            parts.push(format!("Throughput: {throughput}"));
            parts.push(format!("Output size: {output_size}"));
            parts.push(format!("Recall: {recall}"));
            parts.push(format!("Distance computations: {distance_computations}"));
        }

        ftlog::info!("{}", parts.join(", "));
    }

    /// Logs the last k record.
    fn log_last_k(&self) {
        let mut parts = vec![
            format!("Dataset: {}", self.dataset),
            format!("Cardinality: {}", self.cardinality),
            format!("Dimensionality: {}", self.dimensionality),
            format!("Metric: {}", self.metric),
        ];
        if let Some((cluster, algorithm, k, time, throughput, output_size, recall, distance_computations)) =
            self.k_results.last()
        {
            parts.push(format!("Cluster: {cluster}"));
            parts.push(format!("Algorithm: {algorithm}"));
            parts.push(format!("k: {k}"));
            parts.push(format!("Time: {time}"));
            parts.push(format!("Throughput: {throughput}"));
            parts.push(format!("Output size: {output_size}"));
            parts.push(format!("Recall: {recall}"));
            parts.push(format!("Distance computations: {distance_computations}"));
        }

        ftlog::info!("{}", parts.join(", "));
    }

    /// Write the report to a csv file.
    ///
    /// # Errors
    ///
    /// - If the file cannot be created.
    /// - If the header cannot be written.
    /// - If a record cannot be written.
    pub fn write_to_csv<P: AsRef<std::path::Path>>(&self, dir: P) -> Result<(), String> {
        let name = format!(
            "{}_{}_{}_{}.csv",
            self.dataset, self.cardinality, self.dimensionality, self.metric
        );

        let header = [
            "cluster",
            "algorithm",
            "radius",
            "k",
            "time",
            "throughput",
            "mean_output_size",
            "mean_recall",
            "mean_distance_computations",
        ];

        let path = dir.as_ref().join(name);
        let mut writer = csv::Writer::from_path(path).map_err(|e| e.to_string())?;
        writer.write_record(header).map_err(|e| e.to_string())?;

        for (cluster, algorithm, radius, time, throughput, output_size, recall, distance_computations) in
            &self.radial_results
        {
            writer
                .write_record([
                    cluster,
                    algorithm,
                    &radius.to_string(),
                    "",
                    &time.to_string(),
                    &throughput.to_string(),
                    &output_size.to_string(),
                    &recall.to_string(),
                    &distance_computations.to_string(),
                ])
                .map_err(|e| e.to_string())?;
        }

        for (cluster, algorithm, k, time, throughput, output_size, recall, distance_computations) in &self.k_results {
            writer
                .write_record([
                    cluster,
                    algorithm,
                    "",
                    &k.to_string(),
                    &time.to_string(),
                    &throughput.to_string(),
                    &output_size.to_string(),
                    &recall.to_string(),
                    &distance_computations.to_string(),
                ])
                .map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Read the report from a csv file.
    ///
    /// # Errors
    ///
    /// - If the file name is invalid.
    /// - If the record length is invalid.
    /// - If the record parts cannot be parsed.
    pub fn read_from_csv<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let name = path.as_ref().file_stem().ok_or("Could not get file stem")?;
        let name = name.to_string_lossy().to_string().replace(".csv", "");
        let name_parts = name.split('_').collect::<Vec<_>>();
        if name_parts.len() != 4 {
            return Err(format!(
                "Invalid file name. Should have 6 parts separated by underscores: {name_parts:?}"
            ));
        }
        let dataset = name_parts[0].to_string();
        let cardinality = name_parts[1].parse::<usize>().map_err(|e| e.to_string())?;
        let dimensionality = name_parts[2].parse::<usize>().map_err(|e| e.to_string())?;
        let metric = name_parts[3].to_string();

        let mut reader = csv::Reader::from_path(path).map_err(|e| e.to_string())?;
        let mut radial_results = Vec::new();
        let mut k_results = Vec::new();

        for record in reader.records() {
            let record = record.map_err(|e| e.to_string())?;
            if record.len() != 7 {
                return Err(format!("Invalid record length. Should have 7 parts: {record:?}"));
            }
            let cluster = record[0].to_string();
            let algorithm = record[1].to_string();
            let time = record[3].parse::<f32>().map_err(|e| e.to_string())?;
            let throughput = record[4].parse::<f32>().map_err(|e| e.to_string())?;
            let output_size = record[5].parse::<f32>().map_err(|e| e.to_string())?;
            let recall = record[6].parse::<f32>().map_err(|e| e.to_string())?;
            let distance_computations = record[7].parse::<f32>().map_err(|e| e.to_string())?;

            if let Ok(radius) = T::from_str(&record[2]) {
                radial_results.push((
                    cluster,
                    algorithm,
                    radius,
                    time,
                    throughput,
                    output_size,
                    recall,
                    distance_computations,
                ));
            } else if let Ok(k) = record[2].parse::<usize>() {
                k_results.push((
                    cluster,
                    algorithm,
                    k,
                    time,
                    throughput,
                    output_size,
                    recall,
                    distance_computations,
                ));
            } else {
                return Err("Could not parse T or usize from string".to_string());
            }
        }

        Ok(Self {
            dataset,
            cardinality,
            dimensionality,
            metric,
            radial_results,
            k_results,
        })
    }
}
