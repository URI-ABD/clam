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
    depth: usize,
    name: String,
    variant: String,
    radius: f64,
    lfd: f64,
    indices: Option<Vec<usize>>,
    children: Option<[String; 2]>,
    ratios: [f64; 6],
    naive_radius: f64,
    scaled_radius: f64,
}

pub fn report_tree<'a, T, S>(
    dir: &std::path::Path,
    root: &'a clam::Cluster<'a, T, S>,
    build_time: f64,
) -> Result<(), String>
where
    T: clam::Number,
    S: clam::Space<'a, T>,
{
    if !dir.exists() {
        Err(format!("Path does not exist: {dir:?}"))
    } else if !dir.is_dir() {
        Err(format!("Path is not a directory: {dir:?}"))
    } else {
        let report = TreeReport {
            data_name: root.space().data().name(),
            metric_name: root.space().metric().name(),
            cardinality: root.space().data().cardinality(),
            dimensionality: root.space().data().dimensionality(),
            root_name: root.name(),
            max_depth: root.max_leaf_depth(),
            build_time,
        };
        let report = serde_json::to_string_pretty(&report)
            .map_err(|reason| format!("Could not convert report to json because {reason}"))?;

        let path = dir.join("tree.json");
        let mut file = std::fs::File::create(&path)
            .map_err(|reason| format!("Could not create/open file {path:?} because {reason}"))?;
        write!(&mut file, "{report}")
            .map_err(|reason| format!("Could not write report to {path:?} because {reason}."))?;

        let success = root
            .subtree()
            .into_par_iter()
            // .into_iter()
            .map(|c| _report_tree(dir, c))
            .into_par_iter()
            // .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map(|v| v.len());
        match success {
            Ok(num_written) => {
                log::info!("Wrote {num_written} cluster reports.");
                Ok(())
            }
            Err(e) => Err(format!("Failed to write cluster reports because {e}.")),
        }
    }
}

fn _report_tree<'a, T, S>(dir: &std::path::Path, cluster: &'a clam::Cluster<'a, T, S>) -> Result<(), String>
where
    T: clam::Number,
    S: clam::Space<'a, T>,
{
    let report = if cluster.is_leaf() {
        ClusterReport {
            cardinality: cluster.cardinality(),
            depth: cluster.depth(),
            name: cluster.name(),
            variant: cluster.variant_name().to_string(),
            radius: cluster.radius(),
            lfd: cluster.lfd(),
            indices: Some(cluster.indices()),
            children: None,
            ratios: [0.; 6],
            naive_radius: cluster.naive_radius(),
            scaled_radius: cluster.scaled_radius(),
        }
    } else {
        let [left, right] = cluster.children();
        ClusterReport {
            cardinality: cluster.cardinality(),
            depth: cluster.depth(),
            name: cluster.name(),
            variant: cluster.variant_name().to_string(),
            radius: cluster.radius(),
            lfd: cluster.lfd(),
            indices: None,
            children: Some([left.name(), right.name()]),
            ratios: [0.; 6],
            naive_radius: cluster.naive_radius(),
            scaled_radius: cluster.scaled_radius(),
        }
    };

    let report = serde_json::to_string_pretty(&report)
        .map_err(|reason| format!("Could not convert report to json because {reason}"))?;
    let mut path = dir.join(cluster.name());
    path.set_extension("json");
    let mut file = std::fs::File::create(&path)
        .map_err(|reason| format!("Could not create/open file {path:?} because {reason}"))?;
    write!(&mut file, "{report}").map_err(|reason| format!("Could not write report to {path:?} because {reason}."))
}

// #[derive(Debug, Serialize, Deserialize)]
// pub struct RnnReport<'a> {
//     pub data_name: &'a str,
//     pub metric_name: &'a str,
//     pub num_queries: usize,
//     pub num_runs: usize,
//     pub cardinality: usize,
//     pub dimensionality: usize,
//     pub tree_depth: usize,
//     pub build_time: f64,
//     pub root_radius: f64,
//     pub search_radii: Vec<f64>,
//     pub search_times: Vec<Vec<f64>>,
//     pub outputs: Vec<Vec<usize>>,
//     pub recalls: Vec<f64>,
// }

// impl<'a> RnnReport<'a> {
//     #[allow(dead_code)]
//     pub fn is_valid(&self) -> Vec<String> {
//         let mut reasons = vec![];

//         if self.num_queries != self.search_radii.len() {
//             reasons.push(format!(
//                 "self.num_queries != self.search_radii.len(): {} != {}",
//                 self.num_queries,
//                 self.search_radii.len()
//             ));
//         }

//         if self.num_queries != self.search_times.len() {
//             reasons.push(format!(
//                 "self.num_queries != self.search_times.len(): {} != {}",
//                 self.num_queries,
//                 self.search_times.len()
//             ));
//         }

//         if self.num_queries != self.outputs.len() {
//             reasons.push(format!(
//                 "self.num_queries != self.outputs.len(): {} != {}",
//                 self.num_queries,
//                 self.outputs.len()
//             ));
//         }

//         if self.num_queries != self.recalls.len() {
//             reasons.push(format!(
//                 "self.num_queries != self.recalls.len(): {} != {}",
//                 self.num_queries,
//                 self.recalls.len()
//             ));
//         }

//         for (i, samples) in self.search_times.iter().enumerate() {
//             if self.num_runs != samples.len() {
//                 reasons.push(format!(
//                     "{}/{}: self.num_runs != samples.len(): {} != {}",
//                     i + 1,
//                     self.num_queries,
//                     self.num_runs,
//                     samples.len()
//                 ));
//             }
//         }

//         reasons
//     }
// }
