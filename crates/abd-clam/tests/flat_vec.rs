//! Tests for the `FlatVec` struct.

use abd_clam::{
    dataset::{AssociatesMetadata, Permutable},
    Dataset, FlatVec,
};

#[test]
fn creation() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6]];

    let dataset = FlatVec::new(items.clone())?;
    assert_eq!(dataset.cardinality(), 3);
    assert_eq!(dataset.dimensionality_hint(), (0, None));

    let dataset = FlatVec::new_array(items)?;
    assert_eq!(dataset.cardinality(), 3);
    assert_eq!(dataset.dimensionality_hint(), (2, Some(2)));

    Ok(())
}

#[test]
fn ser_de() -> Result<(), String> {
    type Fv = FlatVec<Vec<i32>, usize>;

    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let dataset: Fv = FlatVec::new_array(items)?;

    let serialized: Vec<u8> = bitcode::encode(&dataset).map_err(|e| e.to_string())?;
    let deserialized: Fv = bitcode::decode(&serialized).map_err(|e| e.to_string())?;

    assert_eq!(dataset.cardinality(), deserialized.cardinality());
    assert_eq!(dataset.dimensionality_hint(), deserialized.dimensionality_hint());
    assert_eq!(dataset.permutation(), deserialized.permutation());
    assert_eq!(dataset.metadata(), deserialized.metadata());
    for i in 0..dataset.cardinality() {
        assert_eq!(dataset.get(i), deserialized.get(i));
    }

    Ok(())
}

#[test]
fn permutations() -> Result<(), String> {
    struct SwapTracker {
        data: FlatVec<Vec<i32>, usize>,
        count: usize,
    }

    impl Dataset<Vec<i32>> for SwapTracker {
        fn name(&self) -> &str {
            "SwapTracker"
        }

        fn with_name(mut self, name: &str) -> Self {
            self.data = self.data.with_name(name);
            self
        }

        fn cardinality(&self) -> usize {
            self.data.cardinality()
        }

        fn dimensionality_hint(&self) -> (usize, Option<usize>) {
            self.data.dimensionality_hint()
        }

        fn get(&self, index: usize) -> &Vec<i32> {
            self.data.get(index)
        }
    }

    impl Permutable for SwapTracker {
        fn permutation(&self) -> Vec<usize> {
            self.data.permutation()
        }

        fn set_permutation(&mut self, permutation: &[usize]) {
            self.data.set_permutation(permutation);
        }

        fn swap_two(&mut self, i: usize, j: usize) {
            self.data.swap_two(i, j);
            self.count += 1;
        }
    }

    let items = vec![
        vec![1, 2],
        vec![3, 4],
        vec![5, 6],
        vec![7, 8],
        vec![9, 10],
        vec![11, 12],
    ];
    let data = FlatVec::new_array(items.clone())?;
    let mut swap_tracker = SwapTracker { data, count: 0 };

    swap_tracker.swap_two(0, 2);
    assert_eq!(swap_tracker.permutation(), &[2, 1, 0, 3, 4, 5]);
    assert_eq!(swap_tracker.count, 1);
    for (i, &j) in swap_tracker.permutation().iter().enumerate() {
        assert_eq!(swap_tracker.get(i), &items[j]);
    }

    swap_tracker.swap_two(0, 4);
    assert_eq!(swap_tracker.permutation(), &[4, 1, 0, 3, 2, 5]);
    assert_eq!(swap_tracker.count, 2);
    for (i, &j) in swap_tracker.permutation().iter().enumerate() {
        assert_eq!(swap_tracker.get(i), &items[j]);
    }

    let data = FlatVec::new_array(items.clone())?;
    let mut data = SwapTracker { data, count: 0 };
    let permutation = vec![2, 1, 0, 5, 4, 3];
    data.permute(&permutation);
    assert_eq!(data.permutation(), permutation);
    assert_eq!(data.count, 2);
    for (i, &j) in data.permutation().iter().enumerate() {
        assert_eq!(data.get(i), &items[j]);
    }

    Ok(())
}

#[cfg(feature = "disk-io")]
#[test]
fn npy_io() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let dataset = FlatVec::new_array(items)?;

    let tmp_dir = tempdir::TempDir::new("testing").map_err(|e| e.to_string())?;
    let path = tmp_dir.path().join("test.npy");
    dataset.write_npy(&path)?;

    let new_dataset = FlatVec::<Vec<i32>, _>::read_npy(&path)?;
    assert_eq!(new_dataset.cardinality(), 3);
    assert_eq!(new_dataset.dimensionality_hint(), (2, Some(2)));
    for i in 0..dataset.cardinality() {
        assert_eq!(dataset.get(i), new_dataset.get(i));
    }

    let new_dataset = FlatVec::<Vec<i32>, _>::read_npy(&path)?;
    assert_eq!(new_dataset.cardinality(), 3);
    assert_eq!(new_dataset.dimensionality_hint(), (2, Some(2)));
    for i in 0..dataset.cardinality() {
        assert_eq!(dataset.get(i), new_dataset.get(i));
    }

    let new_dataset = FlatVec::<Vec<i32>, _>::read_npy(&path)?;
    assert_eq!(new_dataset.cardinality(), 3);
    assert_eq!(new_dataset.dimensionality_hint(), (2, Some(2)));
    for i in 0..dataset.cardinality() {
        assert_eq!(dataset.get(i), new_dataset.get(i));
    }

    Ok(())
}
