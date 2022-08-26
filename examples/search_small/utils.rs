use std::io::Write;

// use clam::utils::reports;

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
pub struct BatchReport {
    pub num_queries: usize,
    pub num_runs: usize,
    pub times: Vec<Vec<f64>>,
    pub reports: Vec<reports::SearchReport>,
    pub recalls: Vec<f64>,
}

impl BatchReport {
    pub fn validate(&self) -> Vec<String> {
        let mut failures = vec![];

        if self.num_queries != self.times.len() {
            failures.push(format!(
                "num_queries != num_times: {} != {}",
                self.num_queries,
                self.times.len()
            ));
        }

        if self.num_queries != self.reports.len() {
            failures.push(format!(
                "num_queries != num_outputs: {} != {}",
                self.num_queries,
                self.reports.len()
            ));
        }

        if self.num_queries != self.recalls.len() {
            failures.push(format!(
                "num_queries != num_recalls: {} != {}",
                self.num_queries,
                self.recalls.len()
            ));
        }

        for (i, samples) in self.times.iter().enumerate() {
            if self.num_runs != samples.len() {
                failures.push(format!(
                    "{}/{}: num_runs != num_samples: {} != {}",
                    i + 1,
                    self.num_queries,
                    self.num_runs,
                    samples.len()
                ));
                break;
            }
        }

        failures
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
pub struct RnnReport {
    pub tree: reports::TreeReport,
    pub radii: Vec<f64>,
    pub report: BatchReport,
}

impl RnnReport {
    pub fn validate(&self) -> Vec<String> {
        let mut failures = self.report.validate();

        if self.report.num_queries != self.radii.len() {
            failures.push(format!(
                "num_queries != num_radii: {} != {}",
                self.report.num_queries,
                self.radii.len()
            ));
        }

        failures
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
pub struct KnnReport {
    pub tree: reports::TreeReport,
    pub ks: Vec<usize>,
    pub report: BatchReport,
}

impl KnnReport {
    pub fn validate(&self) -> Vec<String> {
        let mut failures = self.report.validate();

        if self.report.num_queries != self.ks.len() {
            failures.push(format!(
                "num_queries != num_ks: {} != {}",
                self.report.num_queries,
                self.ks.len()
            ));
        }

        failures
    }
}

pub fn make_dir(path: &std::path::Path, empty: bool) -> Result<(), String> {
    if path.exists() {
        if empty {
            std::fs::read_dir(&path).unwrap().into_iter().for_each(|f| {
                std::fs::remove_file(f.unwrap().path()).unwrap();
            });
        }
        Ok(())
    } else {
        std::fs::create_dir(path).map_err(|reason| format!("Could not create directory {:?} because {}.", path, reason))
    }
}

pub fn write_report<R: serde::Serialize>(report: R, path: &std::path::Path) -> Result<(), String> {
    let report = serde_json::to_string_pretty(&report)
        .map_err(|reason| format!("Could not convert report to json because {}.", reason))?;
    let mut file = std::fs::File::create(&path)
        .map_err(|reason| format!("Could not create/open file {:?} because {}.", path, reason))?;
    write!(&mut file, "{}", report)
        .map_err(|reason| format!("Could not write report to {:?} because {}.", path, reason))?;
    Ok(())
}
