use std::io::Write;
use std::path::Path;

use abd_clam::{
    pancakes::{decode_general, encode_general, CodecData, SquishyBall},
    Cluster, PartitionCriteria, VecDataset,
};
use distances::strings::Penalties;
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

        let metadata = dataset.metadata().to_vec();
        let _dataset = CodecData::new(
            root,
            &dataset,
            encode_general::<u16>,
            decode_general,
            metadata,
        )?;

        todo!("Finish implementing the rest of the main function, {expected_name}")

        // let mut data = GenomicDataset::new(base_data, 4, encode_general::<u16>, decode_general);

        // let seed = Some(42);
        // let partition_criteria = PartitionCriteria::default();
        // let root = UniBall::new_root(&data, seed).partition(&mut data, &partition_criteria, seed);
        // let mut root = SquishyBall::from_base_tree(root, &data);
        // root.apply_criteria();
        // let num_clusters = root.subtree().len();
        // root.trim();
        // let num_clusters_after_trim = root.subtree().len();

        // let (data, root) = (data, root);

        // // Write the dataset to a binary file
        // let bin_dir = dataset_dir.join(format!("tree-{n}-{m}"));
        // if !bin_dir.exists() {
        //     std::fs::create_dir(&bin_dir).map_err(|e| e.to_string())?;
        // } else if !bin_dir.is_dir() {
        //     return Err(format!("{bin_dir:?} is not a directory"));
        // }
        // let data_path = bin_dir.join("data.bin");
        // let mut writer = std::fs::File::create(&data_path).map_err(|e| e.to_string())?;
        // data.save(&mut writer, &root)?;
        // writer.flush().map_err(|e| e.to_string())?;
        // let tree_path = bin_dir.join("tree.bin");
        // root.save(&tree_path)?;

        // // Reload the dataset and tree, and check that they are the same
        // let (re_root, re_data) = {
        //     let re_root = SquishyBall::<u16>::load(&tree_path)?;
        //     let mut reader = std::fs::File::open(&data_path).map_err(|e| e.to_string())?;
        //     let re_data = GenomicDataset::<u16, String>::load(
        //         &mut reader,
        //         lev_metric,
        //         true,
        //         encode_general::<u16>,
        //         decode_general,
        //         &re_root,
        //     )?;
        //     (re_root, re_data)
        // };
        // assert_eq!(root.subtree(), re_root.subtree());
        // assert_eq!(data.name(), re_data.name());
        // assert_eq!(data.cardinality(), re_data.cardinality());
        // assert_eq!(data.base_data.metadata(), re_data.base_data.metadata());
        // for i in 0..data.cardinality() {
        //     assert_eq!(
        //         data.base_data.metadata_of(i),
        //         re_data.base_data.metadata_of(i)
        //     );
        //     assert_eq!(data[i], re_data[i]);
        // }

        // for i in 0..data.cardinality() {
        //     assert_eq!(
        //         data.base_data.metadata_of(i),
        //         re_data.base_data.metadata_of(i)
        //     );
        //     assert_eq!(data[i], re_data[i])
        // }

        // // Get the size of the binary file
        // let bin_size = {
        //     data_path.metadata().map_err(|e| e.to_string())?.len()
        //         + tree_path.metadata().map_err(|e| e.to_string())?.len()
        // };

        // // Get the size of the text file
        // let txt_size = expected_path.metadata().map_err(|e| e.to_string())?.len();

        // println!("{expected_name}: Dataset: {}, Root: {re_root}, Clusters: {num_clusters}, Trimmed: {num_clusters_after_trim}", data.name());

        // println!(
        //     "Recursive cost: {}, Unitary cost: {}, Estimated Factor: {:.2e}",
        //     re_root.recursive_cost(),
        //     re_root.unitary_cost(),
        //     re_root.recursive_cost().as_f64() / re_root.unitary_cost().as_f64()
        // );

        // println!(
        //     "File sizes: Text: {txt_size}, Binary: {bin_size}, Actual Factor: {:.2e}",
        //     bin_size.as_f64() / txt_size.as_f64()
        // );

        // // for c in root.subtree().into_iter().filter(|c| c.is_leaf()) {
        // //     println!(
        // //         "Leaf: {}, Depth: {}, Center: {}, Unitary Cost: {}",
        // //         c.name(),
        // //         c.depth(),
        // //         data.base_data.metadata_of(c.arg_center()),
        // //         c.unitary_cost()
        // //     );
        // // }
        // println!();
    }

    Ok(())
}
