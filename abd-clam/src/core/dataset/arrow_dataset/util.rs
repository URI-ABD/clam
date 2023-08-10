use arrow2::{
    array::Float32Array,
    chunk::Chunk,
    datatypes::{DataType::Float32, Field, Schema},
    io::ipc::write::{FileWriter, WriteOptions},
};
use rand::{Rng, SeedableRng};
use std::fs::create_dir;
use std::{fs::File, path::PathBuf};
use uuid::Uuid;

/// Returns the path of the newly created dataset
#[allow(dead_code)]
pub(crate) fn generate_batched_arrow_test_data(
    batches: usize,
    dimensionality: usize,
    cols_per_batch: usize,
    seed: Option<u64>,
    uneven_cols: Option<usize>,
) -> PathBuf {
    // Create a new uuid'd temp directory to store our batches in
    let path = std::env::temp_dir().join(format!("arrow-test-data-{}", Uuid::new_v4()));
    create_dir(path.clone()).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap());

    // Create our fields
    let fields = (0..cols_per_batch)
        .map(|x| Field::new(x.to_string(), Float32, false))
        .collect::<Vec<Field>>();

    // From those fields construct our schema
    let schema = Schema::from(fields);

    // For each batch, we need to create a file, create the write options, create the file writer to write
    // the arrow data, craete our arrays and finally construct our chunk and write it out.
    for batch_number in 0..batches {
        let file = File::create(path.join(format!("batch-{}.arrow", batch_number))).unwrap();
        let options = WriteOptions { compression: None };
        let mut writer = FileWriter::try_new(file, schema.clone(), None, options).unwrap();

        let arrays = (0..cols_per_batch)
            .map(|_| {
                Float32Array::from_vec((0..dimensionality).map(|_| rng.gen_range(0.0..100_000.0)).collect()).boxed()
            })
            .collect();

        let chunk = Chunk::try_new(arrays).unwrap();
        writer.write(&chunk, None).unwrap();
        writer.finish().unwrap();
    }

    if let Some(cols) = uneven_cols {
        // Create our fields
        let fields = (0..cols)
            .map(|x| Field::new(x.to_string(), Float32, false))
            .collect::<Vec<Field>>();

        // From those fields construct our schema
        let schema = Schema::from(fields);
        let file = File::create(path.join(format!("batch-{}.arrow", batches))).unwrap();
        let options = WriteOptions { compression: None };
        let mut writer = FileWriter::try_new(file, schema.clone(), None, options).unwrap();

        let arrays = (0..cols)
            .map(|_| {
                Float32Array::from_vec((0..dimensionality).map(|_| rng.gen_range(0.0..100_000.0)).collect()).boxed()
            })
            .collect();

        let chunk = Chunk::try_new(arrays).unwrap();
        writer.write(&chunk, None).unwrap();
        writer.finish().unwrap();
    }

    // Return the path to our temp directory where the files are stored
    path
}
