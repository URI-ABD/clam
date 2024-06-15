use distances::strings::Penalties;
use std::io::Write;
use std::path::Path;
use symagen::random_edits::{generate_clumped_data, generate_random_string};

/// Writes the given clumped data to a .txt file.
///
/// The file name will be "n-m.txt" where n is the number of clumps and m is the clump size.
/// The file will contain pairs of lines, with the first line being the metadata for the point
/// and the second line being the point itself.
/// The metadata lines are of the form ixj where i is the clump index and j is the index of the point within the clump.
///
/// # Arguments
///
/// `clumped_data`: The clumped data to write to the file.
/// `path`: The directory where the file will be created.
///
/// # Errors
///
/// * If the path does not exist
/// * If the path is not a directory
/// * If `clumped_data` is empty
/// * If the file cannot be created
fn write_data(clumped_data: &[(String, String)], path: &Path) -> Result<(), String> {
    // Check if path exits
    if !path.exists() {
        return Err("Path does not exist".to_string());
    }

    // Check if the path is a directory
    if !path.is_dir() {
        return Err("Path is not a directory".to_string());
    }

    // Check if the clumped data is empty
    if clumped_data.is_empty() {
        return Err("No data to write".to_string());
    }

    let (n, m) = clumped_data
        .last()
        .map(|(x, _)| {
            let mut parts = x.split('x');
            let n = parts.next().unwrap().parse::<usize>().unwrap();
            let m = parts.next().unwrap().parse::<usize>().unwrap();
            (n + 1, m + 1)
        })
        .ok_or("No data to write")?;

    let filename = format!("{n}-{m}.txt");

    let filepath = path.join(filename);

    let mut file = std::fs::File::create(filepath).map_err(|e| e.to_string())?;
    for (x, y) in clumped_data {
        writeln!(file, "{x}").map_err(|e| e.to_string())?;
        writeln!(file, "{y}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 2 {
        return Err(format!(
            "Expected single input parameter, the directory where dataset will be saved. Got {args:?} instead",
        )
        );
    }

    // Parse args[0] into path object
    let dataset_dir = Path::new(&args[1]);
    let dataset_dir = std::fs::canonicalize(dataset_dir).map_err(|e| e.to_string())?;

    // Check that `dataset_dir` has a parent
    if dataset_dir.parent().is_none() {
        return Err("No parent directory".to_string());
    }

    // Check whether the parent of dataset_dir exists
    if !dataset_dir.parent().unwrap().exists() {
        return Err(format!("Parent of {dataset_dir:?} does not exist."));
    }

    // If dataset_dir does not exist, create it. Otherwise, delete all files in it.
    if !dataset_dir.exists() {
        std::fs::create_dir(&dataset_dir).map_err(|e| e.to_string())?;
    } else {
        for entry in std::fs::read_dir(&dataset_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            std::fs::remove_file(entry.path()).map_err(|e| e.to_string())?;
        }
    }

    let alphabet = ['A', 'C', 'G', 'T'];
    let seed_string = generate_random_string(100, &alphabet);
    let penalties: Penalties<u16> = Penalties::new(0, 1, 1);

    for (n, m) in [(16, 16), (16, 32), (32, 16)] {
        let clumped_data = generate_clumped_data(&seed_string, penalties, &alphabet, n, m, 5);
        write_data(&clumped_data, &dataset_dir)?;
    }

    Ok(())
}
