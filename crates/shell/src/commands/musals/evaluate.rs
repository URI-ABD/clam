//! Evaluate MSA quality metrics.

use std::path::Path;

use crate::{
    data::{MusalsSequence, OutputFormat},
    metrics::Metric,
};

/// Evaluate MSA quality metrics.
pub fn evaluate_msa<P: AsRef<Path>>(
    fasta_path_or_dir: P,
    metric: &Metric,
    quality_metrics: &[super::ShellQualityMetric],
    sample_size: Option<usize>,
    out_path: P,
) -> Result<(), String> {
    ftlog::info!("Output path: {:?}", out_path.as_ref());
    let out_dir = out_path
        .as_ref()
        .parent()
        .ok_or_else(|| "Output path must have a parent directory.".to_string())?;

    if fasta_path_or_dir.as_ref().is_dir() {
        let out_name = out_path
            .as_ref()
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "Failed to get output file name.".to_string())?;
        let fasta_files = fasta_path_or_dir.as_ref().read_dir().map_err(|e| e.to_string())?;
        for entry in fasta_files {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_file() {
                let file_ext = path.extension().and_then(|s| s.to_str()).unwrap_or_default().to_lowercase();
                if file_ext == "fasta" {
                    let file_stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .ok_or_else(|| "Failed to get file stem.".to_string())?;
                    let out_path = out_dir.join(format!("{file_stem}-{out_name}"));
                    eval_msa(&path, metric, quality_metrics, sample_size, &out_path)?;
                }
            }
        }
    } else {
        eval_msa(fasta_path_or_dir, metric, quality_metrics, sample_size, out_path)?;
    }

    Ok(())
}

/// Evaluate MSA quality metrics.
fn eval_msa<P: AsRef<Path>>(
    fasta_path: P,
    metric: &Metric,
    quality_metrics: &[super::ShellQualityMetric],
    sample_size: Option<usize>,
    out_path: P,
) -> Result<(), String> {
    let out_path = out_path.as_ref();

    ftlog::info!("Output path: {out_path:?}");
    let _ = out_path.parent().ok_or_else(|| "Output path must have a parent directory.".to_string())?;

    if !matches!(crate::data::InputFormat::from_path(&fasta_path)?, crate::data::InputFormat::Fasta) {
        return Err("MUSALS evaluation currently only supports FASTA input format.".to_string());
    }

    let aligned_items = crate::data::fasta::read(fasta_path, false)?;
    let metric = match metric {
        Metric::Levenshtein => {
            let sz_device = stringzilla::szs::DeviceScope::default().map_err(|e| e.to_string())?;
            let sz_engine = stringzilla::szs::LevenshteinDistances::new(&sz_device, 0, 1, 1, 1).map_err(|e| e.to_string())?;
            move |a: &MusalsSequence, b: &MusalsSequence| -> u32 {
                let distances = sz_engine.compute(&sz_device, &[a.as_ref()], &[b.as_ref()]).unwrap();
                distances[0] as u32
            }
        }
        _ => return Err("MUSALS evaluation currently only supports Levenshtein and Lcs metrics.".to_string()),
    };

    let results = quality_metrics
        .iter()
        .map(|m| {
            let name = m.name();
            ftlog::info!("Preparing quality metric: {name}");
            let result = m.get().par_compute(&aligned_items, &metric, sample_size);
            ftlog::info!("Computed quality metric: {name}");
            (name, result)
        })
        .collect::<Vec<_>>();

    ftlog::info!("Writing all metrics to {out_path:?}...");
    OutputFormat::write_pretty(out_path, &results)
}
