//! A `FlatVec` is a `Dataset` that in which the items are stored in a vector.

use rand::Rng;

use super::{AssociatesMetadata, AssociatesMetadataMut, Dataset, ParDataset, Permutable};

/// A `FlatVec` is a `Dataset` that in which the items are stored in a vector.
///
/// # Type Parameters
///
/// - `I`: The items in the dataset.
/// - `Me`: The metadata associated with the items.
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
pub struct FlatVec<I, Me> {
    /// The items in the dataset.
    items: Vec<I>,
    /// A hint for the dimensionality of the dataset.
    dimensionality_hint: (usize, Option<usize>),
    /// The permutation of the items.
    permutation: Vec<usize>,
    /// The metadata associated with the items.
    pub(crate) metadata: Vec<Me>,
    /// The name of the dataset.
    name: String,
}

impl<I> FlatVec<I, usize> {
    /// Creates a new `FlatVec`.
    ///
    /// The metadata is set to the indices of the items.
    ///
    /// # Errors
    ///
    /// * If the `items` are empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use abd_clam::{Dataset, FlatVec};
    ///
    /// let items = vec![1, 2, 3];
    /// let data = FlatVec::new(items).unwrap();
    /// assert_eq!(data.cardinality(), 3);
    ///
    /// let items: Vec<i32> = vec![];
    /// let data = FlatVec::new(items);
    /// assert!(data.is_err());
    /// ```
    pub fn new(items: Vec<I>) -> Result<Self, String> {
        if items.is_empty() {
            Err("The items are empty.".to_string())
        } else {
            let permutation = (0..items.len()).collect::<Vec<_>>();
            let metadata = permutation.clone();
            Ok(Self {
                items,
                dimensionality_hint: (0, None),
                permutation,
                metadata,
                name: "Unknown FlatVec".to_string(),
            })
        }
    }
}

impl<T> FlatVec<Vec<T>, usize> {
    /// Creates a new `FlatVec` from tabular data.
    ///
    /// The metadata is set to the indices of the items.
    ///
    /// The items are assumed to all have the same length. This length is used
    /// as the dimensionality of the dataset.
    ///
    /// # Errors
    ///
    /// * If the items are empty.
    /// * If the items do not all have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// use abd_clam::{Dataset, FlatVec};
    ///
    /// let items = vec![vec![1, 2], vec![3, 4]];
    /// let data = FlatVec::new_array(items).unwrap();
    /// assert_eq!(data.cardinality(), 2);
    /// ```
    pub fn new_array(items: Vec<Vec<T>>) -> Result<Self, String> {
        if items.is_empty() {
            Err("The items are empty.".to_string())
        } else {
            let (min_len, max_len) = items.iter().fold((usize::MAX, 0), |(min, max), item| {
                (min.min(item.len()), max.max(item.len()))
            });
            if min_len == max_len {
                let permutation = (0..items.len()).collect::<Vec<_>>();
                let metadata = permutation.clone();
                Ok(Self {
                    items,
                    dimensionality_hint: (min_len, Some(min_len)),
                    permutation,
                    metadata,
                    name: "Unknown FlatVec".to_string(),
                })
            } else {
                Err(format!(
                    "The items do not all have the same length. Lengths range from {min_len} to {max_len}."
                ))
            }
        }
    }
}

impl<I, Me> FlatVec<I, Me> {
    /// Sets a lower bound for the dimensionality of the dataset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use abd_clam::{Dataset, FlatVec};
    ///
    /// let items = vec!["hello", "ciao", "classic"];
    /// let data = FlatVec::new(items).unwrap();
    /// assert_eq!(data.dimensionality_hint(), (0, None));
    ///
    /// let data = data.with_dim_lower_bound(3);
    /// assert_eq!(data.dimensionality_hint(), (3, None));
    /// ```
    #[must_use]
    pub const fn with_dim_lower_bound(mut self, lower_bound: usize) -> Self {
        self.dimensionality_hint.0 = lower_bound;
        self
    }

    /// Sets an upper bound for the dimensionality of the dataset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use abd_clam::{Dataset, FlatVec};
    ///
    /// let items = vec!["hello", "ciao", "classic"];
    /// let data = FlatVec::new(items).unwrap();
    /// assert_eq!(data.dimensionality_hint(), (0, None));
    ///
    /// let data = data.with_dim_upper_bound(5);
    /// assert_eq!(data.dimensionality_hint(), (0, Some(5)));
    /// ```
    #[must_use]
    pub const fn with_dim_upper_bound(mut self, upper_bound: usize) -> Self {
        self.dimensionality_hint.1 = Some(upper_bound);
        self
    }

    /// Changes the permutation in the dataset without reordering the items.
    ///
    /// # Example
    ///
    /// ```rust
    /// use abd_clam::{Dataset, FlatVec, dataset::Permutable};
    ///
    /// let items = vec!["hello", "ciao", "classic"];
    /// let data = FlatVec::new(items).unwrap();
    /// assert_eq!(data.permutation(), vec![0, 1, 2]);
    ///
    /// let permutation = vec![1, 0, 2];
    /// let data = data.with_permutation(&permutation);
    /// assert_eq!(data.permutation(), permutation);
    /// ```
    #[must_use]
    pub fn with_permutation(mut self, permutation: &[usize]) -> Self {
        self.set_permutation(permutation);
        self
    }

    /// Get the items in the dataset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use abd_clam::{Dataset, FlatVec};
    ///
    /// let items = vec![1, 2, 3];
    /// let data = FlatVec::new(items).unwrap();
    /// assert_eq!(data.items(), &[1, 2, 3]);
    /// ```
    #[must_use]
    pub fn items(&self) -> &[I] {
        &self.items
    }

    /// Takes the items out of the dataset.
    #[must_use]
    pub fn take_items(self) -> Vec<I> {
        self.items
    }

    /// Transforms the items in the dataset.
    ///
    /// # Type Parameters
    ///
    /// - `It`: The transformed items.
    /// - `F`: The transformer function.
    ///
    /// # Example
    ///
    /// ```rust
    /// use abd_clam::{Dataset, FlatVec};
    ///
    /// let items = vec![1, 2, 3];
    /// let data = FlatVec::new(items).unwrap();
    /// assert_eq!(data.get(0), &1);
    /// assert_eq!(data.get(1), &2);
    /// assert_eq!(data.get(2), &3);
    ///
    /// let f = |x: i32| x * 2;
    /// let data = data.transform_items(f);
    /// assert_eq!(data.get(0), &2);
    /// assert_eq!(data.get(1), &4);
    /// assert_eq!(data.get(2), &6);
    ///
    /// let f = |x: i32| vec![x, x * 2];
    /// let data = data.transform_items(f);
    /// assert_eq!(data.get(0), &[2, 4]);
    /// assert_eq!(data.get(1), &[4, 8]);
    /// assert_eq!(data.get(2), &[6, 12]);
    ///
    /// let f = |x: Vec<i32>| x.into_iter().sum::<i32>().to_string();
    /// let data = data.transform_items(f);
    /// assert_eq!(data.get(0), "6");
    /// assert_eq!(data.get(1), "12");
    /// assert_eq!(data.get(2), "18");
    /// ```
    pub fn transform_items<It, F: Fn(I) -> It>(self, transformer: F) -> FlatVec<It, Me> {
        let items = self.items.into_iter().map(transformer).collect();
        FlatVec {
            items,
            dimensionality_hint: self.dimensionality_hint,
            permutation: self.permutation,
            metadata: self.metadata,
            name: self.name,
        }
    }
}

impl<I: Clone, Me: Clone> FlatVec<I, Me> {
    /// Creates a subsample of the dataset by sampling without replacement.
    ///
    /// This will inherit `dimensionality_hint` from the original dataset. The
    /// permutation will be set to the identity permutation.
    #[must_use]
    pub fn random_subsample<R: Rng>(&self, rng: &mut R, size: usize) -> Self {
        let indices = rand::seq::index::sample(rng, self.items.len(), size).into_vec();
        let items = indices.iter().map(|&i| self.items[i].clone()).collect();
        let metadata = indices.iter().map(|&i| self.metadata[i].clone()).collect();
        Self {
            items,
            dimensionality_hint: self.dimensionality_hint,
            permutation: (0..size).collect(),
            metadata,
            name: self.name.clone(),
        }
    }
}

impl<I, Me> Dataset<I> for FlatVec<I, Me> {
    fn name(&self) -> &str {
        &self.name
    }

    fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    fn cardinality(&self) -> usize {
        self.items.len()
    }

    fn dimensionality_hint(&self) -> (usize, Option<usize>) {
        self.dimensionality_hint
    }

    fn get(&self, index: usize) -> &I {
        &self.items[index]
    }
}

impl<I: Send + Sync, Me: Send + Sync> ParDataset<I> for FlatVec<I, Me> {}

impl<I, Me> AssociatesMetadata<I, Me> for FlatVec<I, Me> {
    fn metadata(&self) -> &[Me] {
        &self.metadata
    }

    fn metadata_at(&self, index: usize) -> &Me {
        &self.metadata[index]
    }
}

impl<I, Me, Met: Clone> AssociatesMetadataMut<I, Me, Met, FlatVec<I, Met>> for FlatVec<I, Me> {
    fn metadata_mut(&mut self) -> &mut [Me] {
        &mut self.metadata
    }

    fn metadata_at_mut(&mut self, index: usize) -> &mut Me {
        &mut self.metadata[index]
    }

    fn with_metadata(self, metadata: &[Met]) -> Result<FlatVec<I, Met>, String> {
        if metadata.len() == self.items.len() {
            let mut metadata = metadata.to_vec();
            metadata.permute(&self.permutation);
            Ok(FlatVec {
                items: self.items,
                dimensionality_hint: self.dimensionality_hint,
                permutation: self.permutation,
                metadata,
                name: self.name,
            })
        } else {
            Err(format!(
                "The metadata length does not match the number of items. {} vs {}",
                metadata.len(),
                self.items.len()
            ))
        }
    }

    fn transform_metadata<F: Fn(&Me) -> Met>(self, f: F) -> FlatVec<I, Met> {
        let metadata = self.metadata.iter().map(f).collect();
        FlatVec {
            items: self.items,
            dimensionality_hint: self.dimensionality_hint,
            permutation: self.permutation,
            metadata,
            name: self.name,
        }
    }
}

impl<I, Me> Permutable for FlatVec<I, Me> {
    fn permutation(&self) -> Vec<usize> {
        self.permutation.clone()
    }

    fn set_permutation(&mut self, permutation: &[usize]) {
        self.permutation = permutation.to_vec();
    }

    fn swap_two(&mut self, i: usize, j: usize) {
        self.items.swap(i, j);
        self.permutation.swap(i, j);
        self.metadata.swap(i, j);
    }
}

#[cfg(feature = "disk-io")]
impl<I: bitcode::Encode + bitcode::Decode, Me: bitcode::Encode + bitcode::Decode> super::DatasetIO<I>
    for FlatVec<I, Me>
{
}

#[cfg(feature = "disk-io")]
impl<I: bitcode::Encode + bitcode::Decode + Send + Sync, Me: bitcode::Encode + bitcode::Decode + Send + Sync>
    super::ParDatasetIO<I> for FlatVec<I, Me>
{
}

#[cfg(feature = "disk-io")]
impl<T: ndarray_npy::ReadableElement + Copy> FlatVec<Vec<T>, usize> {
    /// Reads a `FlatVec` from a `.npy` file.
    ///
    /// The name of the dataset is set to the name of the file without the
    /// extension.
    ///
    /// # Parameters
    ///
    /// - `path`: The path to the `.npy` file.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be read.
    pub fn read_npy<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let name = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let arr: ndarray::Array2<T> = ndarray_npy::read_npy(path)
            .map_err(|e| format!("Could not read npy file: {e}, path: {:?}", path.as_ref()))?;
        let items = arr.axis_iter(ndarray::Axis(0)).map(|row| row.to_vec()).collect();

        Self::new_array(items).map(|data| data.with_name(&name))
    }
}

#[cfg(feature = "disk-io")]
impl<T: ndarray_npy::WritableElement + Copy, Me> FlatVec<Vec<T>, Me> {
    /// Writes the `FlatVec` to a `.npy` file in the given directory.
    ///
    /// # Parameters
    ///
    /// - `path`: The path in which to write the dataset.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be created.
    /// * If the items cannot be converted to an `Array2`.
    /// * If the `Array2` cannot be written.
    pub fn write_npy<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let (min_dim, max_dim) = self.dimensionality_hint;
        let max_dim = max_dim.ok_or_else(|| "Cannot write FlatVec with unknown dimensionality to npy".to_string())?;
        if min_dim != max_dim {
            return Err("Cannot write FlatVec with variable dimensionality to npy".to_string());
        }

        let shape = (self.items.len(), min_dim);
        let v = self.items.iter().flat_map(|row| row.iter().copied()).collect();
        let arr = ndarray::Array2::<T>::from_shape_vec(shape, v)
            .map_err(|e| format!("Could not convert items to Array2: {e}"))?;
        ndarray_npy::write_npy(path, &arr)
            .map_err(|e| e.to_string())
            .map_err(|e| format!("Could not write npy file: {e}, path: {:?}", path.as_ref()))?;

        Ok(())
    }
}

#[cfg(feature = "disk-io")]
impl<T: std::str::FromStr + Copy> FlatVec<Vec<T>, usize> {
    /// Reads a `FlatVec` from a `.csv` file.
    ///
    /// # Parameters
    ///
    /// - `path`: The path to the `.csv` file.
    /// - `delimiter`: The delimiter used in the `.csv` file.
    /// - `has_headers`: Whether to treat the first row as headers.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be read.
    /// * If the types in the file are not parsable as `T`.
    /// * If the items cannot be converted to a `Vec`.
    pub fn read_csv<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let mut reader = csv::ReaderBuilder::new()
            .from_path(path)
            .map_err(|e| format!("Could not start reading csv file: {e}, path: {:?}", path.as_ref()))?;
        let items = reader
            .records()
            .map(|record| {
                record.map_err(|e| e.to_string()).and_then(|record| {
                    record
                        .iter()
                        .map(|field| {
                            field
                                .parse::<T>()
                                .map_err(|_| "Could not parse field in csv".to_string())
                        })
                        .collect()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new_array(items)
    }
}

#[cfg(feature = "disk-io")]
impl<T: std::string::ToString + Copy, M> FlatVec<Vec<T>, M> {
    /// Writes the `FlatVec` to a `.csv` file with the given path.
    ///
    /// # Parameters
    ///
    /// - `path`: The path to the `.csv` file.
    /// - `delimiter`: The delimiter to use in the `.csv` file.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be created.
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: &P, delimiter: u8) -> Result<(), String> {
        let mut writer = csv::WriterBuilder::new()
            .delimiter(delimiter)
            .from_path(path)
            .map_err(|e| format!("Could not start csv file: {e}, path: {:?}", path.as_ref()))?;
        for item in &self.items {
            writer
                .write_record(item.iter().map(T::to_string))
                .map_err(|e| format!("Could not write record to csv: {e}"))?;
        }
        Ok(())
    }
}
