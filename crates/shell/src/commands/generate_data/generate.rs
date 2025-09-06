//! Implementation of dataset generation logic.

use std::path::Path;

use crate::data::{DataType, InputFormat, MusalsSequence, ShellData};

/// Generate a synthetic dataset with the specified parameters.
///
/// # Arguments
///
/// * `num_vectors` - Total number of vectors to generate (m)
/// * `dimensions` - Dimensionality of each vector (n)
/// * `out_path` - Output file path with extension indicating format
/// * `data_type` - The data type for generated vectors
/// * `partitions` - Optional partition splits as percentages (e.g., vec![95, 5])
/// * `min_val` - Minimum value for numeric data
/// * `max_val` - Maximum value for numeric data
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * `Ok(())` if generation succeeds
/// * `Err(String)` with an error message if generation fails
///
/// # Examples
///
/// ```ignore
/// let mut rng = rand::rng();
///
/// // Generate 100 12-dimensional f32 vectors, split 95/5
/// generate_dataset(
///     100,
///     12,
///     PathBuf::from("../data/small-vectors.npy"),
///     DataType::F32,
///     Some(vec![95, 5]),
///     0.0,
///     1.0,
///     &mut rng,
/// )?;
/// // This creates: data-95.npy and data-5.npy
/// ```
#[allow(clippy::too_many_arguments)]
pub fn generate_dataset<P: AsRef<Path> + core::fmt::Debug, R: rand::Rng>(
    num_vectors: usize,
    dimensions: usize,
    out_path: P,
    data_type: DataType,
    partitions: Option<Vec<usize>>,
    min_val: f64,
    max_val: f64,
    rng: &mut R,
) -> Result<(), String> {
    // Validate partitions sum to 100 if provided
    if let Some(ref parts) = partitions {
        let sum: usize = parts.iter().sum();
        if sum != 100 {
            return Err(format!("Partition percentages must sum to 100, got: {sum}"));
        }
        if parts.is_empty() {
            return Err("At least one partition must be specified".to_string());
        }
    }

    let fmt = InputFormat::from_path(&out_path).map_err(|e| format!("Failed to determine format from file extension: {e}"))?;
    match (&fmt, &data_type) {
        (InputFormat::Fasta, DataType::String) => {} // Valid
        (InputFormat::Fasta, _) => {
            return Err("Fasta format only supports String data type".to_string());
        }
        (_, DataType::String) => {
            return Err("String data type only supports Fasta format".to_string());
        }
        _ => {} // Other combinations are valid
    }

    ftlog::info!("=== Generating Dataset ===");
    ftlog::info!("Generating {num_vectors} {dimensions}-dimensional vectors of type {data_type:?}");
    ftlog::info!("Output filename: {out_path:?}");
    ftlog::info!("Partitions: {partitions:?}");
    ftlog::info!("Value range: [{min_val}, {max_val}]");
    ftlog::info!("");

    // Generate the data based on type
    let shell_data = match data_type {
        DataType::F32 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as f32, max_val as f32, rng);
            ShellData::F32(data)
        }
        DataType::F64 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val, max_val, rng);
            ShellData::F64(data)
        }
        DataType::I8 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as i8, max_val as i8, rng);
            ShellData::I8(data)
        }
        DataType::I16 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as i16, max_val as i16, rng);
            ShellData::I16(data)
        }
        DataType::I32 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as i32, max_val as i32, rng);
            ShellData::I32(data)
        }
        DataType::I64 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as i64, max_val as i64, rng);
            ShellData::I64(data)
        }
        DataType::U8 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as u8, max_val as u8, rng);
            ShellData::U8(data)
        }
        DataType::U16 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as u16, max_val as u16, rng);
            ShellData::U16(data)
        }
        DataType::U32 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as u32, max_val as u32, rng);
            ShellData::U32(data)
        }
        DataType::U64 => {
            let data = symagen::random_data::random_tabular(num_vectors, dimensions, min_val as u64, max_val as u64, rng);
            ShellData::U64(data)
        }
        DataType::String => generate_strings(num_vectors, dimensions, rng),
    };

    // Write data to file(s) based on partitions
    if let Some(parts) = partitions {
        write_partitions(&shell_data, out_path, num_vectors, &parts)?;
    } else {
        shell_data.write(&out_path)?;
        ftlog::info!("✓ Successfully wrote {out_path:?}");
    }

    ftlog::info!("");
    ftlog::info!("✓ Dataset generation complete!");

    Ok(())
}

/// Generate random string sequences and wrap them in a ShellData.
fn generate_strings<R: rand::Rng>(num_vectors: usize, avg_length: usize, rng: &mut R) -> ShellData {
    // Use DNA alphabet for biological sequences
    let alphabet = "ACGT";
    let min_len = avg_length.saturating_sub(avg_length / 4);
    let max_len = avg_length + avg_length / 4;

    let data = symagen::random_data::random_string(num_vectors, min_len, max_len, alphabet, rng);
    let metadata = (0..num_vectors).map(|i| format!(">seq_{}", i + 1)).collect::<Vec<String>>();

    let data = metadata.into_iter().zip(data.into_iter().map(MusalsSequence)).collect();

    ShellData::String(data)
}

/// Write partitioned data to multiple files.
fn write_partitions<P: AsRef<Path> + core::fmt::Debug>(data: &ShellData, out_path: P, total_vectors: usize, partitions: &[usize]) -> Result<(), String> {
    let out_path = out_path.as_ref();
    let out_dir = out_path.parent().ok_or_else(|| "Output path must have a parent directory".to_string())?;
    let filename = out_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Output file must have a valid name".to_string())?;
    let ext = out_path
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Output file must have a valid extension".to_string())?;

    let mut start_idx = 0;

    for (idx, &percentage) in partitions.iter().enumerate() {
        // Calculate the number of vectors for this partition
        let count = (total_vectors * percentage) / 100;

        // Handle rounding issues: if this is the last partition, take all remaining vectors
        let count = if idx == partitions.len() - 1 { total_vectors - start_idx } else { count };

        if count == 0 {
            return Err(format!(
                "Partition {idx} ({percentage} percent) results in 0 vectors. Total vectors: {total_vectors}",
            ));
        }

        let end_idx = start_idx + count;

        // Create a slice of the data
        let partition = data.slice(start_idx, end_idx);

        // Write the partition to a file
        let partition_path = out_dir.join(format!("{filename}-{percentage}.{ext}"));
        ftlog::info!("Writing {count} vectors to: {partition_path:?}");
        partition.write(&partition_path)?;
        ftlog::info!("✓ Successfully wrote {partition_path:?}");

        start_idx = end_idx;
    }

    Ok(())
}
