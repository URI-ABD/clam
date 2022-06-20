use std::io::Read;

use clam::prelude::*;

pub static SEARCH_1B_NAMES: &[[&str; 4]] = &[[
    "msft_spacev",
    "msft_spacev-1b.i8bin",
    "msft_spacev-query.i8bin",
    "msft_spacev-ground.bin",
]];

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

    let num_points = u32::from_le_bytes(read_one(&mut reader, 4)?.try_into().unwrap());
    let num_dimensions = u32::from_le_bytes(read_one(&mut reader, 4)?.try_into().unwrap());
    let instance_size = num_dimensions * T::num_bytes() as u32;

    println!(
        "num_points: {}, num_dimensions: {}, instance_size {}.",
        num_points, num_dimensions, instance_size
    );

    let data: Vec<Vec<T>> = {
        let data = read_one(&mut reader, (num_points * instance_size) as u64)?;
        data.chunks(instance_size as usize)
            .map(|row_bytes| {
                row_bytes
                    .chunks(T::num_bytes() as usize)
                    .map(|bytes| T::from_le_bytes(bytes).unwrap())
                    .collect()
            })
            .collect()
    };
    // for i in 0..num_points {
    //     let instance = read_one(&mut reader, instance_size)?;
    //     let instance = instance.chunks(T::num_bytes() as usize).map(|bytes| T::from_le_bytes(bytes).unwrap()).collect();
    //     data.push(instance);

    //     if i % 1000 == 0 {
    //         println!("Read {} instances ...", i);
    //     }
    // }

    println!("Read {} instances", data.len());

    Ok(data)
}

fn read_ground(path: &std::path::PathBuf) -> std::io::Result<(Data<u32>, Data<f32>)> {
    println!("{:?}", path);
    todo!()
}

pub fn read_data<T: Number>(index: usize) -> (Data<T>, Data<T>, Data<u32>, Data<f32>) {
    assert!(
        index < SEARCH_1B_NAMES.len(),
        "index must be smaller than {}. Got {} instead.",
        SEARCH_1B_NAMES.len(),
        index
    );

    let [dir, train_name, query_name, ground_name] = SEARCH_1B_NAMES[index];

    let mut data_dir = std::path::PathBuf::from("/");
    data_dir.push("data");
    data_dir.push("abd");
    data_dir.push("search_data");
    data_dir.push(dir);

    let train_data = {
        let mut train_path = data_dir.clone();
        train_path.push(train_name);
        assert!(train_path.exists(), "{:?} does not exist.", &train_path);
        read_instances(&train_path).unwrap()
    };

    let query_data = {
        let mut query_path = data_dir.clone();
        query_path.push(query_name);
        assert!(query_path.exists(), "{:?} does not exist.", &query_path);
        read_instances(&query_path).unwrap()
    };

    let (neighbors, distances) = {
        let mut ground_path = data_dir.clone();
        ground_path.push(ground_name);
        assert!(ground_path.exists(), "{:?} does not exist.", &ground_path);
        read_ground(&ground_path).unwrap()
    };

    (train_data, query_data, neighbors, distances)
}
