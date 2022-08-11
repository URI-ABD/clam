#[allow(dead_code)]
pub struct H5Data<Tr: crate::h5number::H5Number> {
    data: hdf5::Dataset,
    name: String,
    shape: (usize, usize),
    zero: Tr,
}

impl<Tr: crate::h5number::H5Number> H5Data<Tr> {
    pub fn new(file: &hdf5::File, member_name: &str, name: String) -> Result<Self, String> {
        let data = file
            .dataset(member_name)
            .map_err(|reason| format!("Could not read member {} because {}", member_name, reason))?;
        let shape = data.shape();
        Ok(Self {
            data,
            name,
            shape: (shape[0], shape[1]),
            zero: Tr::zero(),
        })
    }

    pub fn to_vec_vec<T>(&self) -> Result<Vec<Vec<T>>, String>
    where
        T: clam::Number,
    {
        let data: ndarray::Array2<Tr> = self
            .data
            .read_2d()
            .map_err(|reason| format!("Could not convert from HDF% to Tabular because {}", reason))?;

        Ok(data
            .outer_iter()
            .map(|row| row.into_iter().map(|&v| T::from(v).unwrap()).collect())
            .collect())
    }
}

impl<Tr: crate::h5number::H5Number> std::fmt::Debug for H5Data<Tr> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Dataset")
            .field("name", &self.name)
            .field("cardinality", &self.shape.0)
            .field("dimensionality", &self.shape.1)
            .finish()
    }
}

impl<Tr: crate::h5number::H5Number, T: clam::Number> clam::Dataset<T> for H5Data<Tr> {
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
        let row: Vec<Tr> = self.data.read_slice_1d(ndarray::s![index, ..]).unwrap().to_vec();
        row.into_iter().map(|v| T::from(v).unwrap()).collect()
    }
}
