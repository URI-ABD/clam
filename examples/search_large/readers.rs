use serde::Serialize;
use std::io::Read;

use super::chunked_data::Data;

type Reader = std::io::BufReader<std::fs::File>;

pub struct BigAnnPaths<'a> {
    pub folder: &'a str,
    pub train: &'a str,
    pub subset_1: Option<&'a str>,
    pub subset_2: Option<&'a str>,
    pub subset_3: Option<&'a str>,
    pub query: &'a str,
    pub ground: &'a str,
}

pub fn make_dir(path: &std::path::Path) -> Result<(), String> {
    if !path.exists() {
        std::fs::create_dir(path).map_err(|reason| format!("Could not create directory {path:?} because {reason}."))
    } else {
        Ok(())
    }
}

fn read_row<R: std::io::Read>(reader: &mut R, row_size: usize) -> Result<Vec<u8>, String> {
    let mut buffer = vec![];
    let mut chunk = reader.take(row_size as u64);
    let n = chunk
        .read_to_end(&mut buffer)
        .map_err(|reason| format!("Could not read chunk from reader because {reason}."))?;
    if n == row_size {
        Ok(buffer)
    } else {
        Err(format!(
            "Could not read the correct number of bytes. Expected {row_size} but got {n}."
        ))
    }
}

fn read_table<T: clam::Number, R: std::io::Read>(
    reader: &mut R,
    num_rows: usize,
    bytes_per_row: usize,
) -> Result<Data<T>, String> {
    let mut data = Data::new();
    for _ in 0..num_rows {
        let row = read_row(reader, bytes_per_row)?;
        let row = row
            .chunks(T::num_bytes() as usize)
            .map(|bytes| T::from_le_bytes(bytes).unwrap())
            .collect();
        data.push(row);
    }
    Ok(data)
}

pub fn open_reader(path: &std::path::Path) -> Result<Reader, String> {
    let reader = std::io::BufReader::new(
        std::fs::File::open(path).map_err(|reason| format!("Could not open file {path:?} because {reason}"))?,
    );
    Ok(reader)
}

fn transform_train<T: clam::Number>(in_path: &std::path::Path, out_dir: &std::path::Path) -> Result<(), String> {
    let mut reader = open_reader(in_path)?;

    let num_points = u32::from_le_bytes(read_row(&mut reader, 4)?.try_into().unwrap()) as usize;
    let num_dimensions = u32::from_le_bytes(read_row(&mut reader, 4)?.try_into().unwrap()) as usize;
    let instance_size = num_dimensions * T::num_bytes() as usize;

    let chunk_size = 1_000_000;
    let num_chunks = (num_points / chunk_size) + if (num_points % chunk_size) == 0 { 0 } else { 1 };
    let last_chunk_size = if (num_points % chunk_size) == 0 {
        chunk_size
    } else {
        num_points % chunk_size
    };
    for (chunk_index, _) in (1..=num_points).step_by(chunk_size).enumerate() {
        println!("Transforming chunk {}/{} ...", chunk_index + 1, num_chunks);

        let num_rows = if chunk_index == (num_chunks - 1) {
            last_chunk_size
        } else {
            chunk_size
        };
        let chunk = read_table::<T, Reader>(&mut reader, num_rows, instance_size)?;

        let mut buffer = vec![];
        chunk.serialize(&mut rmp_serde::Serializer::new(&mut buffer)).unwrap();

        std::fs::write(out_dir.join(format!("chunk_{chunk_index}_{num_rows}")), buffer)
            .map_err(|reason| format!("Could not write point {chunk_index} because {reason}."))?
    }

    Ok(())
}

pub fn transform<T: clam::Number>(
    data: &BigAnnPaths,
    in_dir: &std::path::Path,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    make_dir(out_dir)?;

    let train_out = out_dir.join("train");
    make_dir(&train_out)?;

    transform_train::<T>(&in_dir.join(data.train), &train_out)?;

    Ok(())
}
