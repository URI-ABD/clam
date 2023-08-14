#[cfg(test)]
mod tests {
    use crate::core::dataset::{BatchedArrowDataset, Dataset};
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

    /// Generates a new random f32 arrow dataset and returns the directory in which it was created (/tmp/)
    /// Notably, this function can create datasets with final batches that have an arbitrarily smaller
    /// number of columns than other batches. This allows for testing unevenly split datasets.
    ///
    /// # Args
    /// - `batches` - The number of batches in the dataset
    /// - `dimensionality` - The dimensionality of the dataset
    /// - `cols_per_batch` - The number of columns per batch (I.e. rows in a normal format)
    /// - `seed` - The seed for the rng
    /// - `uneven_cols` - If this is `Some(n)`, then the last batch will have `n` columns. Note that `n`
    /// must be less than or equal to `cols_per_batch`.
    ///
    /// # Returns
    /// The path where the dataset was generated.
    pub fn generate_batched_arrow_test_data(
        batches: usize,
        dimensionality: usize,
        cols_per_batch: usize,
        seed: Option<u64>,
        uneven_cols: Option<usize>,
    ) -> PathBuf {
        // Create a new uuid'd temp directory to satore our batches in
        let path = std::env::temp_dir().join(format!("arrow-test-data-{}", Uuid::new_v4()));
        create_dir(path.clone()).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap());

        // Create our fields
        let fields = (0..cols_per_batch)
            .map(|x| Field::new(x.to_string(), Float32, false))
            .collect::<Vec<Field>>();

        // From those fields construct our schema
        let schema = Schema::from(fields);

        let batches = if uneven_cols.is_some() { batches - 1 } else { batches };

        // For each batch, we need to create a file, create the write options, create the file writer to write
        // the arrow data, craete our arrays and finally construct our chunk and write it out.
        for batch_number in 0..batches {
            let file = File::create(path.join(format!("batch-{batch_number}.arrow"))).unwrap();
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
            let file = File::create(path.join(format!("batch-{batches}.arrow"))).unwrap();
            let options = WriteOptions { compression: None };
            let mut writer = FileWriter::try_new(file, schema, None, options).unwrap();

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

    fn euclidean(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        distances::vectors::euclidean(a, b)
    }

    fn assert_approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 0.01);
    }

    #[test]
    fn grab_col_raw() {
        let batches = 100;
        let cols_per_batch = 10000;
        let dimensionality = 10;
        let seed = 25565;

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed), None);

        let name = "Test Dataset".to_string();
        let dataset: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        assert_eq!(dataset.cardinality(), batches * cols_per_batch);
        println!("{:?}", dataset.get(0));
        assert_eq!(dataset.get(0).len(), dimensionality);

        // Try to get every column. If we can't then this will panic and the test will fail
        for i in 0..dataset.cardinality() {
            dataset.get(i);
        }
    }

    // Tests the difference between our implementation and the arrow2 implementation
    #[test]
    fn test_diff() {
        let dimensionality = 50;
        let cols_per_batch = 500;

        let path = generate_batched_arrow_test_data(1, dimensionality, cols_per_batch, Some(42), None);
        let mut reader = std::fs::File::open(path.join("batch-0.arrow")).unwrap();
        let metadata = arrow2::io::ipc::read::read_file_metadata(&mut reader).unwrap();
        let mut reader = arrow2::io::ipc::read::FileReader::new(reader, metadata, None, None);

        let binding = reader.next().unwrap().unwrap();
        let columns = binding.columns();

        let name = "Test Dataset".to_string();
        let data: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        for i in 0..cols_per_batch {
            let col: Vec<f32> = columns[i]
                .as_any()
                .downcast_ref::<arrow2::array::PrimitiveArray<f32>>()
                .unwrap()
                .iter()
                .map(|x| *x.unwrap())
                .collect();

            for j in 0..dimensionality {
                assert_approx_eq(col[j], data.get(i)[j]);
            }
        }
    }

    #[test]
    fn test_reorder() {
        let dimensionality = 1;
        let cols_per_batch = 500;

        let path = generate_batched_arrow_test_data(1, dimensionality, cols_per_batch, Some(42), None);
        let name = "Test Dataset".to_string();
        let mut data: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        let reordering = (0..cols_per_batch).rev().collect::<Vec<usize>>();
        data.reorder(&reordering);
        assert_eq!(data.reordered_indices(), reordering);
    }

    #[test]
    fn test_uneven() {
        let batches = 3;
        let cols_per_batch = 4;
        let dimensionality = 4;
        let seed = 25565;
        let uneven = 3;

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed), Some(uneven));

        let name = "Test Dataset".to_string();
        let dataset: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        assert_eq!(
            dataset.cardinality(),
            batches * cols_per_batch - (cols_per_batch - uneven)
        );
        assert_eq!(dataset.get(0).len(), dimensionality);

        for i in 0..dataset.cardinality() {
            dataset.get(i);
        }
    }

    #[test]
    fn test_uneven_correctness() {
        let batches = 3;
        let cols_per_batch = 4;
        let dimensionality = 4;
        let seed = 25565;
        let uneven = 3;

        let expected_cardinality = batches * cols_per_batch - (cols_per_batch - uneven);

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed), Some(uneven));

        let name = "Test Dataset".to_string();
        let dataset: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        assert_eq!(dataset.cardinality(), expected_cardinality);
        assert_eq!(dataset.get(0).len(), dimensionality);

        let mut reader = std::fs::File::open(path.join("batch-2.arrow")).unwrap();
        let metadata = arrow2::io::ipc::read::read_file_metadata(&mut reader).unwrap();
        let mut reader = arrow2::io::ipc::read::FileReader::new(reader, metadata, None, None);

        let binding = reader.next().unwrap().unwrap();
        let columns = binding.columns();

        let offset = expected_cardinality - uneven;
        for i in 0..3 {
            let col: Vec<f32> = columns[i]
                .as_any()
                .downcast_ref::<arrow2::array::PrimitiveArray<f32>>()
                .unwrap()
                .iter()
                .map(|x| *x.unwrap())
                .collect();

            for j in 0..dimensionality {
                assert_approx_eq(col[j], dataset.get(i + offset)[j]);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_oob() {
        let batches = 3;
        let cols_per_batch = 4;
        let dimensionality = 4;
        let seed = 25565;

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed), Some(3));

        let name = "Test Dataset".to_string();
        let dataset: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        dataset.get(15);
    }
}
