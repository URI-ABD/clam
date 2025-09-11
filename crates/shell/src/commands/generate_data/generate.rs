//! Implementation of dataset generation logic.

use crate::data::{DataType, Format, ShellFlatVec};
use abd_clam::{Dataset, FlatVec};

/// Generate a synthetic dataset with the specified parameters.
///
/// # Arguments
///
/// * `num_vectors` - Total number of vectors to generate (m)
/// * `dimensions` - Dimensionality of each vector (n)
/// * `filename` - Base filename for output files (without extension)
/// * `data_type` - The data type for generated vectors
/// * `format` - Output format
/// * `partitions` - Optional partition splits as percentages (e.g., vec![95, 5])
/// * `min_val` - Minimum value for numeric data
/// * `max_val` - Maximum value for numeric data
///
/// # Returns
///
/// * `Ok(())` if generation succeeds
/// * `Err(String)` with an error message if generation fails
///
/// # Examples
///
/// ```ignore
/// // Generate 100 12-dimensional f32 vectors, split 95/5
/// generate_dataset(
///     100,
///     12,
///     "data".to_string(),
///     DataType::F32,
///     Format::Npy,
///     Some(vec![95, 5]),
///     0.0,
///     1.0,
/// )?;
/// // This creates: data-95.npy and data-5.npy
/// ```
#[allow(clippy::too_many_arguments)]
pub fn generate_dataset(
    num_vectors: usize,
    dimensions: usize,
    filename: String,
    data_type: DataType,
    format: Format,
    partitions: Option<Vec<usize>>,
    min_val: f64,
    max_val: f64,
    seed: Option<u64>,
) -> Result<(), String> {
    let seed = seed.unwrap_or(42);
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

    // String type is only valid for FASTA format
    if matches!(data_type, DataType::String) && format != Format::Fasta {
        return Err("String data type is only supported with FASTA format".to_string());
    }

    // FASTA format is only valid for String type
    if format == Format::Fasta && !matches!(data_type, DataType::String) {
        return Err("FASTA format is only supported with String data type".to_string());
    }

    println!("=== Generating Dataset ===");
    println!("Generating {num_vectors} {dimensions}-dimensional vectors of type {data_type:?}");
    println!("Output filename: {filename}");
    println!("Format: {format}");
    println!("Partitions: {partitions:?}");
    println!("Value range: [{min_val}, {max_val}]");
    println!("Seed: {seed}");
    println!();

    // Generate the data based on type
    let shell_flat_vec = match data_type {
        DataType::F32 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as f32,
                max_val as f32,
                seed,
            );
            ShellFlatVec::F32(FlatVec::from_nested_vec(data)?)
        }
        DataType::F64 => {
            let data = symagen::random_data::random_tabular_seedable(num_vectors, dimensions, min_val, max_val, seed);
            ShellFlatVec::F64(FlatVec::from_nested_vec(data)?)
        }
        DataType::I8 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as i8,
                max_val as i8,
                seed,
            );
            ShellFlatVec::I8(FlatVec::from_nested_vec(data)?)
        }
        DataType::I16 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as i16,
                max_val as i16,
                seed,
            );
            ShellFlatVec::I16(FlatVec::from_nested_vec(data)?)
        }
        DataType::I32 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as i32,
                max_val as i32,
                seed,
            );
            ShellFlatVec::I32(FlatVec::from_nested_vec(data)?)
        }
        DataType::I64 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as i64,
                max_val as i64,
                seed,
            );
            ShellFlatVec::I64(FlatVec::from_nested_vec(data)?)
        }
        DataType::U8 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as u8,
                max_val as u8,
                seed,
            );
            ShellFlatVec::U8(FlatVec::from_nested_vec(data)?)
        }
        DataType::U16 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as u16,
                max_val as u16,
                seed,
            );
            ShellFlatVec::U16(FlatVec::from_nested_vec(data)?)
        }
        DataType::U32 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as u32,
                max_val as u32,
                seed,
            );
            ShellFlatVec::U32(FlatVec::from_nested_vec(data)?)
        }
        DataType::U64 => {
            let data = symagen::random_data::random_tabular_seedable(
                num_vectors,
                dimensions,
                min_val as u64,
                max_val as u64,
                seed,
            );
            ShellFlatVec::U64(FlatVec::from_nested_vec(data)?)
        }
        DataType::String => generate_strings(num_vectors, dimensions, seed)?,
    };

    // Write data to file(s) based on partitions
    if let Some(parts) = partitions {
        write_partitions(&shell_flat_vec, &filename, &format, num_vectors, &parts)?;
    } else {
        let output_path = format!("{filename}.{format}");
        println!("Writing {num_vectors} vectors to: {output_path}");
        shell_flat_vec.write(&output_path)?;
        println!("✓ Successfully wrote {output_path}");
    }

    println!();
    println!("✓ Dataset generation complete!");

    Ok(())
}

/// Generate random string sequences and wrap them in a ShellFlatVec.
fn generate_strings(num_vectors: usize, avg_length: usize, seed: u64) -> Result<ShellFlatVec, String> {
    // Use DNA alphabet for biological sequences
    let alphabet = "ACGT";
    let min_len = avg_length.saturating_sub(avg_length / 4);
    let max_len = avg_length + avg_length / 4;

    let data = symagen::random_data::random_string(num_vectors, min_len, max_len, alphabet, seed);

    let flat_vec = FlatVec::new(data)?;
    Ok(ShellFlatVec::String(flat_vec))
}

/// Write partitioned data to multiple files.
fn write_partitions(
    data: &ShellFlatVec,
    filename: &str,
    format: &Format,
    total_vectors: usize,
    partitions: &[usize],
) -> Result<(), String> {
    let mut start_idx = 0;

    for (idx, &percentage) in partitions.iter().enumerate() {
        // Calculate the number of vectors for this partition
        let count = (total_vectors * percentage) / 100;

        // Handle rounding issues: if this is the last partition, take all remaining vectors
        let count = if idx == partitions.len() - 1 {
            total_vectors - start_idx
        } else {
            count
        };

        if count == 0 {
            return Err(format!(
                "Partition {idx} ({percentage} percent) results in 0 vectors. Total vectors: {total_vectors}",
            ));
        }

        let end_idx = start_idx + count;

        // Create a slice of the data
        let partition = slice_shell_flat_vec(data, start_idx, end_idx)?;

        // Write the partition to a file
        let output_path = format!("{filename}-{count}.{format}");
        println!("Writing {count} vectors to: {output_path}");
        partition.write(&output_path)?;
        println!("✓ Successfully wrote {output_path}");

        start_idx = end_idx;
    }

    Ok(())
}

/// Create a slice of a ShellFlatVec from start to end indices.
fn slice_shell_flat_vec(data: &ShellFlatVec, start: usize, end: usize) -> Result<ShellFlatVec, String> {
    match data {
        ShellFlatVec::F32(fv) => {
            let sliced: Vec<Vec<f32>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::F32(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::F64(fv) => {
            let sliced: Vec<Vec<f64>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::F64(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::I8(fv) => {
            let sliced: Vec<Vec<i8>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::I8(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::I16(fv) => {
            let sliced: Vec<Vec<i16>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::I16(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::I32(fv) => {
            let sliced: Vec<Vec<i32>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::I32(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::I64(fv) => {
            let sliced: Vec<Vec<i64>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::I64(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::U8(fv) => {
            let sliced: Vec<Vec<u8>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::U8(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::U16(fv) => {
            let sliced: Vec<Vec<u16>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::U16(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::U32(fv) => {
            let sliced: Vec<Vec<u32>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::U32(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::U64(fv) => {
            let sliced: Vec<Vec<u64>> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::U64(FlatVec::from_nested_vec(sliced)?))
        }
        ShellFlatVec::String(fv) => {
            let sliced: Vec<String> = (start..end).map(|i| fv.get(i).clone()).collect();
            Ok(ShellFlatVec::String(FlatVec::new(sliced)?))
        }
    }
}
