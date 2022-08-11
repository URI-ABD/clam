use std::io::Write;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct TreeReport {
    data_name: String,
    metric_name: String,
    cardinality: usize,
    dimensionality: usize,
    root_name: String,
    max_depth: usize,
    build_time: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClusterReport {
    cardinality: usize,
    indices: Option<Vec<usize>>,
    name: String,
    arg_center: usize,
    arg_radius: usize,
    radius: f64,
    lfd: f64,
    left_child: Option<String>,
    right_child: Option<String>,
    // ratios: [f64; 6],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RnnReport<'a> {
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
    pub outputs: Vec<Vec<usize>>,
    pub recalls: Vec<f64>,
}

pub fn report_tree<'a, T, U>(
    dir: &std::path::Path,
    root: &'a clam::Cluster<'a, T, U>,
    build_time: f64,
) -> Result<(), String>
where
    T: clam::Number,
    U: clam::Number,
{
    if !dir.exists() {
        std::fs::create_dir(dir).map_err(|reason| format!("Could not create dir {:?} because {}", dir, reason))
    } else if !dir.is_dir() {
        Err(format!("Path is not a directory: {:?}", dir))
    } else {
        let report = TreeReport {
            data_name: root.space().data().name(),
            metric_name: root.space().metric().name(),
            cardinality: root.space().data().cardinality(),
            dimensionality: root.space().data().dimensionality(),
            root_name: root.name_str(),
            max_depth: root.max_leaf_depth(),
            build_time,
        };
        let report = serde_json::to_string_pretty(&report)
            .map_err(|reason| format!("Could not convert report to json because {}", reason))?;

        let dir = dir.join(root.space().name());
        if dir.exists() {
            // for entry in std::fs::read_dir(dir)? {}
            std::fs::read_dir(&dir).unwrap().into_iter().for_each(|f| {
                std::fs::remove_file(f.unwrap().path()).unwrap();
            });
        } else {
            std::fs::create_dir(&dir)
                .map_err(|reason| format!("Could create directory {:?} because {}", &dir, reason))?;
        }

        let path = dir.join("tree.json");
        let mut file = std::fs::File::create(&path)
            .map_err(|reason| format!("Could not create/open file {:?} because {}", path, reason))?;
        write!(&mut file, "{}", report)
            .map_err(|reason| format!("Could not write report to {:?} because {}.", path, reason))?;

        root.subtree().into_par_iter().map(|c| _report_tree(&dir, c)).collect()
    }
}

fn _report_tree<'a, T, U>(dir: &std::path::Path, cluster: &'a clam::Cluster<'a, T, U>) -> Result<(), String>
where
    T: clam::Number,
    U: clam::Number,
{
    let report = if cluster.is_leaf() {
        ClusterReport {
            cardinality: cluster.cardinality(),
            indices: Some(cluster.indices()),
            name: cluster.name_str(),
            arg_center: cluster.arg_center(),
            arg_radius: cluster.arg_radius(),
            radius: cluster.radius().as_f64(),
            lfd: cluster.lfd(),
            left_child: None,
            right_child: None,
        }
    } else {
        ClusterReport {
            cardinality: cluster.cardinality(),
            indices: None,
            name: cluster.name_str(),
            arg_center: cluster.arg_center(),
            arg_radius: cluster.arg_radius(),
            radius: cluster.radius().as_f64(),
            lfd: cluster.lfd(),
            left_child: Some(cluster.left_child().name_str()),
            right_child: Some(cluster.right_child().name_str()),
        }
    };

    let report = serde_json::to_string_pretty(&report)
        .map_err(|reason| format!("Could not convert report to json because {}", reason))?;
    let mut path = dir.join(cluster.name_str());
    path.set_extension("json");
    let mut file = std::fs::File::create(&path)
        .map_err(|reason| format!("Could not create/open file {:?} because {}", path, reason))?;
    write!(&mut file, "{}", report).map_err(|reason| format!("Could not write report to {:?} because {}.", path, reason))
}

impl<'a> RnnReport<'a> {
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

        if self.num_queries != self.outputs.len() {
            reasons.push(format!(
                "self.num_queries != self.outputs.len(): {} != {}",
                self.num_queries,
                self.outputs.len()
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
