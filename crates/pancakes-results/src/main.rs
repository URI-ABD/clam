use abd_clam::Dataset;
use abd_clam::VecDataset;
use distances::strings::levenshtein_custom;
use distances::strings::Penalties;
use symagen::random_edits::{generate_clumped_data, generate_random_string};

fn main() -> Result<(), String> {
    let alphabet = ['A', 'C', 'G', 'T'];
    let seed_string = generate_random_string(100, &alphabet);
    let penalties: Penalties<u16> = Penalties::new(0, 1, 1);

    #[allow(clippy::ptr_arg)]
    fn levenshtein(x: &String, y: &String) -> u16 {
        let penalties = Penalties::new(0, 1, 1);
        let levenshtein = levenshtein_custom(penalties);
        levenshtein(x, y)
    }

    let is_metric_expensive = false;

    let dataset_dir = {
        let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
        dir.push("datasets");
        if !dir.exists() {
            std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
        }
        dir
    };

    let clumped_data = generate_clumped_data(&seed_string, penalties, &alphabet, 16, 16, 10);

    let clumped_dataset = VecDataset::new(
        "clumped_dataset".to_string(),
        clumped_data,
        levenshtein,
        is_metric_expensive,
    );

    clumped_dataset
        .save(&dataset_dir.join("clumped_dataset.txt"))
        .unwrap();

    let clumped_data_dense = generate_clumped_data(&seed_string, penalties, &alphabet, 16, 32, 10);

    let clumped_dataset_dense = VecDataset::new(
        "clumped_dataset_dense".to_string(),
        clumped_data_dense,
        levenshtein,
        is_metric_expensive,
    );

    clumped_dataset_dense
        .save(&dataset_dir.join("clumped_dataset_dense.txt"))
        .unwrap();

    let clumped_data_big = generate_clumped_data(&seed_string, penalties, &alphabet, 32, 16, 10);

    let clumped_dataset_big = VecDataset::new(
        "clumped_dataset_big".to_string(),
        clumped_data_big,
        levenshtein,
        is_metric_expensive,
    );

    clumped_dataset_big
        .save(&dataset_dir.join("clumped_dataset_big.txt"))
        .unwrap();

    Ok(())
}
