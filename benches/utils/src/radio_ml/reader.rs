//! Reading the `RadioML` dataset.

use ndarray::prelude::*;

use crate::Complex;

/// Returns the 26 SNR levels in the `RadioML` dataset.
///
/// These are (-20..=30) dB in steps of 2 dB.
fn snr_levels() -> [i32; 26] {
    (-20..=30)
        .step_by(2)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("We have the correct number of iterations."))
}

/// Reads a single modulation mode from the `RadioML` dataset.
///
/// # Arguments
///
/// * `inp_dir` - The input directory containing the `RadioML` dataset.
/// * `mode` - The modulation mode to read.
/// * `snr` - The SNR level to read. If `None`, all SNR levels are read.
///
/// # Returns
///
/// A 2D array with shape `[n_samples, n_features]` containing the signals for
/// the given modulation mode and SNR level.
///
/// # Errors
///
/// - If the file does not exist.
/// - If the dataset does not exist.
/// - If the dataset has the wrong shape.
/// - If the dataset cannot be read.
/// - If the SNR level is not found.
/// - If the dataset has the wrong shape.
pub fn read_mod<P: AsRef<std::path::Path>>(
    inp_dir: &P,
    mode: &super::ModulationMode,
    snr: Option<i32>,
) -> Result<Vec<Vec<Complex<f64>>>, String> {
    let h5_path = mode.h5_path(inp_dir);
    if !h5_path.exists() {
        return Err(format!("File {h5_path:?} does not exist!"));
    }

    let file = hdf5::File::open(&h5_path).map_err(|e| format!("Error opening file: {e}"))?;
    ftlog::debug!("Opened file {h5_path:?}: {file:?}");

    let data_raw = file.dataset("X").map_err(|e| format!("Error opening dataset: {e}"))?;
    ftlog::debug!("Opened dataset X: {data_raw:?}");

    let train = data_raw
        .read_dyn::<f64>()
        .map_err(|e| format!("Error reading dataset: {e}"))?;
    ftlog::debug!("Read dataset X with shape: {:?}", train.shape());

    // `train` should be a 3D array with shape `[106_496, 1_024, 2]`. Each
    // sample is a 2D array with shape `[1_024, 2]` containing the real and
    // imaginary parts of the signal.
    if train.ndim() != 3 {
        return Err(format!("Expected 3D array, got {}D array!", train.ndim()));
    }
    if train.shape() != [106_496, 1_024, 2] {
        return Err(format!("Expected shape [106_496, 1_024, 2], got {:?}!", train.shape()));
    }

    // Convert the array from a dynamic array to a 3D array.
    let train = train
        .into_shape_with_order([106_496, 1_024, 2])
        .map_err(|e| format!("Error reshaping dataset: {e}"))?;

    // The samples are stored in chunks of 4_096 samples, each corresponding to
    // a different SNR level. We need to extract the samples corresponding to
    // the given SNR level if it is provided.
    let train = if let Some(snr) = snr {
        ftlog::debug!("Extracting samples for SNR level {snr}.");
        let snr_idx = snr_levels()
            .into_iter()
            .position(|x| x == snr)
            .ok_or_else(|| format!("SNR level {snr} not found!"))?;
        let start = snr_idx * 4_096;
        let end = start + 4_096;
        train.slice(s![start..end, .., ..]).to_owned()
    } else {
        train
    };
    ftlog::debug!("Extracted samples for SNR level: {:?}", train.shape());

    // For each sample, take the magnitude of the complex number and convert the
    // full array to a 2D array with shape `[106_496, 1_024]`.
    let train = train
        .outer_iter()
        .map(|sample| {
            sample
                .rows()
                .into_iter()
                .map(|r| Complex::<f64>::from((r[0], r[1])))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    ftlog::debug!(
        "Converted dataset from complex to real: {:?}",
        (train.len(), train[0].len())
    );

    ftlog::info!("Read {} signals for {}.", train.len(), mode.name());

    Ok(train)
}
