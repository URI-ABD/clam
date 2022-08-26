use std::io::Read;

use clam::prelude::*;

pub struct BigAnnData<'a> {
    pub folder: &'a str,
    pub train: &'a str,
    pub subset_1: Option<&'a str>,
    pub subset_2: Option<&'a str>,
    pub subset_3: Option<&'a str>,
    pub query: &'a str,
    pub ground: &'a str,
}

pub static BIG_ANN_DATA: &[BigAnnData<'static>] = &[
    BigAnnData {
        folder: "msft_spacev",
        train: "msft_spacev-1b.i8bin",
        subset_1: None,
        subset_2: None,
        subset_3: None,
        query: "public_query_gt100.bin",
        ground: "msspacev-1B",
    },
    BigAnnData {
        folder: "yandex_t2i",
        train: "yandex_t2i-1b.fbin",
        subset_1: Some("base.1M.fbin"),
        subset_2: None,
        subset_3: None,
        query: "yandex_t2i-ground.bin",
        ground: "yandex_t2i-query.fbin",
    },
    BigAnnData {
        folder: "msft_spacev",
        train: "msft_spacev-1b.i8bin",
        subset_1: None,
        subset_2: None,
        subset_3: None,
        query: "msft_spacev-query.i8bin",
        ground: "msft_spacev-ground.bin",
    },
];

pub type Data<T> = Vec<Vec<T>>;

fn read_one<R: std::io::Read>(reader: R, instance_size: u64) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![];
    let mut chunk = reader.take(instance_size);
    // Do appropriate error handling for your situation
    // Maybe it's OK if you didn't read enough bytes?
    let n = chunk.read_to_end(&mut buf)?;
    assert_eq!(instance_size as usize, n);
    Ok(buf)
}

fn read_instances<T: Number>(path: &std::path::PathBuf) -> std::io::Result<Data<T>> {
    println!("Reading from {:?} ...", path);

    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);

    let num_points = u32::from_le_bytes(read_one(&mut reader, 4)?.try_into().unwrap()) as usize;
    let num_dimensions = u32::from_le_bytes(read_one(&mut reader, 4)?.try_into().unwrap()) as usize;
    let instance_size = num_dimensions * T::num_bytes() as usize;

    // let num_bytes = reader.bytes().skip(200_000_000 * instance_size).count();
    // println!("{} instances, {} leftovers ...", num_bytes / instance_size, num_bytes % instance_size);
    // reader.bytes().step_by(1_000_000 * instance_size).enumerate().for_each(|(i, _)| println!("step {} ...", i));

    println!(
        "num_points: {}, num_dimensions: {}, instance_size {}.",
        num_points, num_dimensions, instance_size
    );

    let mut data: Vec<Vec<T>> = vec![];
    if num_points >= 1_000_000_000 {
        let step = 1_000_000;
        for i in 0..num_points {
            let instance = read_one(&mut reader, instance_size as u64)?;
            let instance = instance
                .chunks(T::num_bytes() as usize)
                .map(|bytes| T::from_le_bytes(bytes).unwrap())
                .collect();
            data.push(instance);

            if (i + 1) % step == 0 {
                println!(
                    "Read {}/{} chunks of size {} ...",
                    (i + 1) / step,
                    num_points / step,
                    step
                );
                data.clear();
            }
        }
    } else {
        for _ in 0..num_points {
            let instance = read_one(&mut reader, instance_size as u64)?;
            let instance = instance
                .chunks(T::num_bytes() as usize)
                .map(|bytes| T::from_le_bytes(bytes).unwrap())
                .collect();
            data.push(instance);
        }
        println!("Read {} instances ...", data.len())
    }

    Ok(data)
}

fn read_ground(path: &std::path::PathBuf) -> std::io::Result<(Data<usize>, Data<f32>)> {
    println!("Reading from {:?} ...", path);

    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);

    let num_queries = u32::from_le_bytes(read_one(&mut reader, 4)?.try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(read_one(&mut reader, 4)?.try_into().unwrap()) as usize;

    // Read true indices
    let row_size = k * u32::num_bytes() as usize;

    println!("num_queries: {}, k: {}, row_size {}.", num_queries, k, row_size);

    let mut indices: Vec<Vec<usize>> = vec![];
    for _ in 0..num_queries {
        let instance = read_one(&mut reader, row_size as u64)?;
        let instance = instance
            .chunks(u32::num_bytes() as usize)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
            .collect();
        indices.push(instance);
    }
    println!("Read {} rows of distances ...", indices.len());

    // read true distances ...
    let row_size = k * f32::num_bytes() as usize;

    println!("num_queries: {}, k: {}, row_size {}.", num_queries, k, row_size);

    let mut distances: Vec<Vec<f32>> = vec![];
    for _ in 0..num_queries {
        let instance = read_one(&mut reader, row_size as u64)?;
        let instance = instance
            .chunks(f32::num_bytes() as usize)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        distances.push(instance);
    }
    println!("Read {} rows of distances ...", distances.len());

    Ok((indices, distances))
}

pub fn read_data<T: Number>(index: usize) -> (Data<T>, Data<T>, Data<usize>, Data<f32>) {
    assert!(
        index < BIG_ANN_DATA.len(),
        "index must be smaller than {}. Got {} instead.",
        BIG_ANN_DATA.len(),
        index
    );

    let data = &BIG_ANN_DATA[index];

    let mut data_dir = std::env::current_dir().unwrap();
    data_dir.pop();
    data_dir.push("data");
    data_dir.push("search_large");
    data_dir.push(data.folder);

    // let train_data = vec![];
    let train_data = {
        let mut train_path = data_dir.clone();
        train_path.push(data.train);
        assert!(train_path.exists(), "{:?} does not exist.", &train_path);
        read_instances(&train_path).unwrap()
    };

    // let query_data = vec![];
    let query_data = {
        let mut query_path = data_dir.clone();
        query_path.push(data.query);
        assert!(query_path.exists(), "{:?} does not exist.", &query_path);
        read_instances(&query_path).unwrap()
    };

    let (neighbors, distances) = {
        let mut ground_path = data_dir.clone();
        ground_path.push(data.ground);
        assert!(ground_path.exists(), "{:?} does not exist.", &ground_path);
        read_ground(&ground_path).unwrap()
    };

    (train_data, query_data, neighbors, distances)
}
