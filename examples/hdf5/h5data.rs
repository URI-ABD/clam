pub struct H5Data {
    data: hdf5::Dataset,
    name: String,
    shape: (usize, usize),
}

impl H5Data {
    pub fn new(file: &hdf5::File, member_name: &str, name: String) -> Result<Self, String> {
        let data = file
            .dataset(member_name)
            .map_err(|reason| format!("Could not read member {} because {}", member_name, reason))?;
        let shape = data.shape();
        Ok(Self {
            data,
            name,
            shape: (shape[0], shape[1]),
        })
    }

    pub fn to_vec_vec<T, U>(&self) -> Result<Vec<Vec<U>>, String>
    where
        T: crate::h5number::H5Number,
        U: clam::Number,
    {
        let data: ndarray::Array2<T> = self
            .data
            .read_2d()
            .map_err(|reason| format!("Could not convert from HDF% to Tabular because {}", reason))?;

        Ok(data
            .outer_iter()
            .map(|row| row.into_iter().map(|&v| U::from(v).unwrap()).collect())
            .collect())
    }
}

impl std::fmt::Debug for H5Data {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Dataset")
            .field("name", &self.name)
            .field("cardinality", &self.shape.0)
            .field("dimensionality", &self.shape.1)
            .finish()
    }
}

impl<T: crate::h5number::H5Number> clam::Dataset<T> for H5Data {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn cardinality(&self) -> usize {
        self.shape.0
    }

    fn dimensionality(&self) -> usize {
        self.shape.1
    }

    fn indices(&self) -> Vec<usize> {
        (0..self.shape.0).collect()
    }

    fn get(&self, index: usize) -> Vec<T> {
        self.data.read_slice_1d(ndarray::s![index, ..]).unwrap().to_vec()
    }
}
