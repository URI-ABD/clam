#[cfg(test)]
mod tests {
    use crate::{
        core::dataset::arrow_dataset::util::generate_batched_arrow_test_data,
        core::dataset::{BatchedArrowDataset, Dataset},
    };

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

    // #[test]
    // fn test_cluster() {
    //     let batches = 1;
    //     let cols_per_batch = 4;
    //     let dimensionality = 3;
    //     let seed = 25565;

    //     let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed), None);

    //     let name = "Test Dataset".to_string();
    //     let data =
    //         BatchedArrowDataset::new(path.to_str().unwrap(), name, distances::vectors::euclidean, false).unwrap();

    //     let indices = data.indices().to_vec();
    //     let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
    //     let cluster = Cluster::new_root(&data, &indices, Some(42)).partition(&data, &partition_criteria, true);

    //     assert_eq!(cluster.depth(), 0);
    //     assert_eq!(cluster.cardinality(), 4);
    //     assert_eq!(cluster.num_descendants(), 6);
    //     assert!(cluster.radius() > 0.);
    //     assert_eq!(format!("{cluster}"), "1");

    //     let [left, right] = cluster.children().unwrap();
    //     assert_eq!(format!("{left}"), "2");
    //     assert_eq!(format!("{right}"), "3");
    // }

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

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed), Some(3));

        let name = "Test Dataset".to_string();
        let dataset: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        assert_eq!(dataset.cardinality(), batches * cols_per_batch + 3);
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

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed), Some(uneven));

        let name = "Test Dataset".to_string();
        let dataset: BatchedArrowDataset<f32, f32> =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, euclidean, false).unwrap();

        assert_eq!(dataset.cardinality(), batches * cols_per_batch + 3);
        assert_eq!(dataset.get(0).len(), dimensionality);

        let mut reader = std::fs::File::open(path.join("batch-3.arrow")).unwrap();
        let metadata = arrow2::io::ipc::read::read_file_metadata(&mut reader).unwrap();
        let mut reader = arrow2::io::ipc::read::FileReader::new(reader, metadata, None, None);

        let binding = reader.next().unwrap().unwrap();
        let columns = binding.columns();

        let offset = batches * cols_per_batch;
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
