use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use distances::Number;
use ndarray::prelude::*;
use ndarray_npy::WritableElement;
use num_traits::Zero;
use sysinfo::SystemExt;

pub fn convert_bigann() -> Result<(), String> {
    let [raw_dir, out_dir] = raw_std_dir();

    let base_name = "bigann-1B.u8bin";
    let query_name = "bigann-query.u8bin";
    let cardinality = 1_000_000_000;
    let dimensionality = 128;
    let mut bigann = BigAnnDataset::new(
        "bigann",
        raw_dir,
        out_dir,
        base_name,
        query_name,
        cardinality,
        dimensionality,
    );
    bigann.convert::<u8>(0.25).unwrap();

    Ok(())
}

fn raw_std_dir() -> [PathBuf; 2] {
    let mut data_dir = std::env::current_dir().unwrap();
    data_dir.pop();
    data_dir.push("data");
    assert!(data_dir.exists(), "Path not found: {data_dir:?}");

    let mut raw_dir = data_dir.clone();
    raw_dir.push("raw");
    raw_dir.push("bigann");
    assert!(raw_dir.exists(), "Path not found: {raw_dir:?}");

    let mut std_dir = data_dir;
    std_dir.push("standard");
    std_dir.push("bigann");
    if !std_dir.exists() {
        std::fs::create_dir_all(&std_dir).unwrap();
    }

    [raw_dir, std_dir]
}

pub struct BigAnnDataset {
    name: String,
    base_path: PathBuf,
    query_path: PathBuf,
    out_dir: PathBuf,
    cardinality: usize,
    dimensionality: usize,
    shard_paths: Vec<PathBuf>,
    num_queries: usize,
    query_paths: Vec<PathBuf>,
}

impl BigAnnDataset {
    fn new(
        name: &str,
        raw_dir: PathBuf,
        out_dir: PathBuf,
        base_name: &str,
        query_name: &str,
        cardinality: usize,
        dimensionality: usize,
    ) -> Self {
        let mut base_path = raw_dir.clone();
        base_path.push("base");
        base_path.push(base_name);
        assert!(base_path.exists(), "Path not found: {base_path:?}");

        let mut query_path = raw_dir;
        query_path.push("query");
        query_path.push(query_name);
        assert!(query_path.exists(), "Path not found: {query_path:?}");

        Self {
            name: name.to_string(),
            base_path,
            query_path,
            out_dir,
            cardinality,
            dimensionality,
            shard_paths: Vec::new(),
            num_queries: 0,
            query_paths: Vec::new(),
        }
    }

    pub fn convert<T: Number + Zero + WritableElement>(
        &mut self,
        mem_fraction: f32,
    ) -> Result<(), String> {
        let mem_fraction = if !(0.0..=0.5).contains(&mem_fraction) {
            0.25
        } else {
            mem_fraction
        };

        let available_memory = sysinfo::System::new_all().available_memory();
        println!("Available memory: {available_memory} bytes");

        let data_size = self.cardinality * self.dimensionality * T::num_bytes();
        println!("Data size: {data_size} bytes");

        let available_memory = (available_memory.as_f32() * mem_fraction) as usize;
        let batch_size = available_memory / (self.dimensionality * T::num_bytes());
        let num_batches = self.cardinality / batch_size;

        println!("Converting {} dataset with batch size {batch_size} and expected {num_batches} batches ...", self.name);

        let (_, shard_paths) =
            convert_vectors::<T>(&self.base_path, batch_size, "base", &self.out_dir)?;

        self.shard_paths = shard_paths;

        let ([num_queries, _], query_paths) =
            convert_vectors::<T>(&self.query_path, batch_size, "query", &self.out_dir)?;

        self.num_queries = num_queries;
        self.query_paths = query_paths;

        Ok(())
    }
}

fn convert_vectors<T: Number + Zero + WritableElement>(
    inp_path: &Path,
    batch_size: usize,
    name: &str,
    out_dir: &Path,
) -> Result<([usize; 2], Vec<PathBuf>), String> {
    let mut handle = File::open(inp_path).map_err(|reason| reason.to_string())?;
    let (expected_cardinality, dimensionality) = {
        let car_dim = read_row::<_, u32>(&mut handle, 2)?;
        (car_dim[0] as usize, car_dim[1] as usize)
    };
    println!("Expecting to read {expected_cardinality} points in {dimensionality} dimensions from {name} set ...");

    let mut cardinality = 0;
    let mut shard_paths = Vec::new();
    for (i, _) in (0..expected_cardinality).step_by(batch_size).enumerate() {
        let (batch_len, shard_path) =
            convert_shard::<_, T>(&mut handle, batch_size, dimensionality, out_dir, name, i)?;
        cardinality += batch_len;
        shard_paths.push(shard_path);
    }
    if expected_cardinality == cardinality {
        Ok(([cardinality, dimensionality], shard_paths))
    } else {
        Err(format!(
            "Unable to read the correct number of points. Got {cardinality} but expected {expected_cardinality}."
        ))
    }
}

fn convert_shard<R: Read, T: Number + Zero + WritableElement>(
    handle: &mut R,
    shard_size: usize,
    dimensionality: usize,
    out_dir: &Path,
    name: &str,
    i: usize,
) -> Result<(usize, PathBuf), String> {
    println!("Converting batch {i} from {name} set to npy format ...");

    let mut batch = Array2::zeros((0, dimensionality));
    for _ in 0..shard_size {
        if let Ok(row) = read_row::<R, T>(handle, dimensionality) {
            let row = ArrayView::from(&row);
            batch.push_row(row).map_err(|reason| reason.to_string())?;
        } else {
            break;
        };
    }
    let out_path = {
        let mut path = out_dir.to_owned();
        path.push(format!("{name}-shard-{i}.npy"));
        path
    };
    ndarray_npy::write_npy(&out_path, &batch).map_err(|reason| reason.to_string())?;
    Ok((batch.nrows(), out_path))
}

fn read_row<R: Read, T: Number>(handle: &mut R, dim: usize) -> Result<Vec<T>, String> {
    let num_bytes = T::num_bytes() * dim;
    let mut row = vec![0_u8; num_bytes];

    handle
        .read_exact(&mut row)
        .map_err(|reason| format!("Could not read row from file because {:?}", reason))?;

    let row = row
        .chunks_exact(T::num_bytes())
        .map(|bytes| T::from_le_bytes(bytes))
        .collect();

    Ok(row)
}
