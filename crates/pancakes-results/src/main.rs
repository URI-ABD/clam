use std::io::Write;
use std::path::Path;

use abd_clam::{
    pancakes::{decode_general, encode_general, CodecData, SquishyBall},
    Cluster, PartitionCriteria, VecDataset,
};
use distances::{strings::Penalties, Number};
use symagen::random_edits::{generate_clumped_data, generate_random_string};

#[allow(clippy::ptr_arg)]
fn lev_metric(x: &String, y: &String) -> u16 {
    distances::strings::levenshtein(x, y)
}

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

fn read_data(path: &Path) -> Result<Vec<(String, String)>, String> {
    let lines = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let lines = lines.lines().map(|x| x.to_string()).collect::<Vec<_>>();

    Ok(lines
        .chunks(2)
        .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
        .collect::<Vec<_>>())
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

    // If dataset_dir does not exist, create it.
    if !dataset_dir.exists() {
        std::fs::create_dir(&dataset_dir).map_err(|e| e.to_string())?;
    }

    let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect::<Vec<_>>();
    let seed_string = generate_random_string(1024, &alphabet);
    let penalties = Penalties::<u16>::new(0, 1, 1);

    let sizes = [
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (32, 64),
        (32, 128),
        (32, 256),
        (32, 512),
        (32, 1024),
    ];
    for &(n, m) in &sizes[..] {
        let expected_name = format!("{n}-{m}.txt");
        let expected_path = dataset_dir.join(&expected_name);
        let clumped_data = if !expected_path.exists() {
            let clumped_data = generate_clumped_data(&seed_string, penalties, &alphabet, n, m, 10);
            write_data(&clumped_data, &dataset_dir)?;
            clumped_data
        } else {
            read_data(&expected_path)?
        };

        let (metadata, clumped_data) = clumped_data.into_iter().unzip();

        let name = format!("{n}x{m}");
        let mut dataset =
            VecDataset::new(name, clumped_data, lev_metric, true).assign_metadata(metadata)?;
        let criteria = PartitionCriteria::default();
        let seed = Some(42);
        let root = SquishyBall::new_root(&dataset, seed).partition(&mut dataset, &criteria, seed);
        let tree_size = root.subtree().len();

        let metadata = dataset.metadata().to_vec();
        let dataset = CodecData::new(
            root,
            &dataset,
            encode_general::<u16>,
            decode_general,
            metadata,
        )?;

        // Write the dataset to a binary file
        let bin_dir = dataset_dir.join(format!("codec-{n}-{m}"));
        dataset.save(&bin_dir)?;

        // Reload the dataset and tree, and check that they are the same
        let re_data = CodecData::<String, u16, String>::load(
            &bin_dir,
            lev_metric,
            true,
            encode_general::<u16>,
            decode_general,
        )?;
        assert_eq!(dataset.root().subtree(), re_data.root().subtree());
        assert_eq!(dataset.centers(), re_data.centers());
        assert_eq!(dataset.metadata(), re_data.metadata());

        for (&c, &rc) in dataset
            .root()
            .subtree()
            .iter()
            .zip(re_data.root().subtree().iter())
        {
            assert_eq!(c, rc);
            assert_eq!(c.arg_center(), rc.arg_center());
            assert_eq!(c.arg_radial(), rc.arg_radial());
            assert_eq!(c.radius(), rc.radius());
            assert_eq!(c.arg_poles(), rc.arg_poles());
            assert_eq!(c.recursive_cost(), rc.recursive_cost());
            assert_eq!(c.unitary_cost(), rc.unitary_cost());
        }

        // Get the size of the binary files in the codec directory
        let bin_size = bin_dir
            .read_dir()
            .map_err(|e| e.to_string())?
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .sum::<u64>();

        // Get the size of the text file
        let txt_size = expected_path.metadata().map_err(|e| e.to_string())?.len();

        let trimmed_size = dataset.root().compressible_subtree().len();

        println!(
            "{expected_name}, Root: {}, Clusters: {tree_size}, Trimmed: {trimmed_size}",
            dataset.root().name()
        );

        println!(
            "Recursive cost: {}, Unitary cost: {}, Estimated Factor: {:.2e}",
            dataset.root().recursive_cost(),
            dataset.root().unitary_cost(),
            dataset.root().recursive_cost().as_f64() / dataset.root().unitary_cost().as_f64()
        );

        println!(
            "File sizes: Text: {txt_size}, Binary: {bin_size}, Actual Factor: {:.2e}",
            bin_size.as_f64() / txt_size.as_f64()
        );

        println!();
    }

    Ok(())
}
