use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Deserialize, Serialize)]
pub struct Data<T> {
    data: Vec<Vec<T>>,
}

#[allow(dead_code)]
impl<T: clam::Number> Data<T> {
    pub fn new() -> Self {
        Self { data: vec![] }
    }

    pub fn push(&mut self, row: Vec<T>) {
        self.data.push(row);
    }
}

#[allow(dead_code)]
pub struct ChunkedTabular<'a, T: clam::Number> {
    location: &'a std::path::Path,
    cardinality: usize,
    dimensionality: usize,
    chunk_size: usize,
    num_chunks: usize,
    last_chunk_size: usize,
    name: &'a str,
    zero: T,
}

impl<'a, T: clam::Number> std::fmt::Debug for ChunkedTabular<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Chunked Tabular Dataset")
            .field("name", &self.name)
            .field("cardinality", &self.cardinality)
            .field("dimensionality", &self.dimensionality)
            .field("chunk_size", &self.chunk_size)
            .field("location", &self.location)
            .finish()
    }
}

impl<'a, T: clam::Number> ChunkedTabular<'a, T> {
    pub fn new(
        location: &'a std::path::Path,
        cardinality: usize,
        dimensionality: usize,
        chunk_size: usize,
        name: &'a str,
    ) -> Self {
        let num_chunks = (cardinality / chunk_size) + if (cardinality % chunk_size) == 0 { 0 } else { 1 };
        let last_chunk_size = if (cardinality % chunk_size) == 0 {
            chunk_size
        } else {
            cardinality % chunk_size
        };
        Self {
            location,
            cardinality,
            dimensionality,
            chunk_size,
            num_chunks,
            last_chunk_size,
            name,
            zero: T::zero(),
        }
    }

    #[allow(unused_variables)]
    pub fn get(&self, index: usize) -> &'a [T] {
        let chunk_id = index / self.chunk_size;
        let chunk_size = if chunk_id == (self.num_chunks - 1) {
            self.last_chunk_size
        } else {
            self.chunk_size
        };
        let name = format!("chunk_{chunk_id}_{chunk_size}");
        let path = self.location.join(name);

        let chunk = {
            let mut reader = rmp_serde::Deserializer::new(super::readers::open_reader(&path).unwrap());
            Data::<T>::deserialize(&mut reader).unwrap()
        };

        // chunk.data[index % self.chunk_size].clone();
        todo!()
    }
}

impl<'a, T: clam::Number> clam::Dataset<'a, T> for ChunkedTabular<'a, T> {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn dimensionality(&self) -> usize {
        self.dimensionality
    }

    fn indices(&self) -> Vec<usize> {
        (0..self.cardinality).collect()
    }

    fn get(&self, index: usize) -> &'a [T] {
        self.get(index)
    }
}
