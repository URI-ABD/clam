//! Read and write FASTA datasets.

use bio::io::fasta;

/// Reads a FASTA file from the given path.
pub fn read<P: AsRef<std::path::Path>>(path: &P, remove_gaps: bool) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
    let reader = fasta::Reader::from_file(path.as_ref()).map_err(|e| e.to_string())?;

    if remove_gaps {
        ftlog::info!("Removing gaps from sequences while reading FASTA file: {:?}", path.as_ref());
    }

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let id = record.id().to_string();
        let seq = if remove_gaps {
            let gaps = [b'-', b'.']; // Common gap characters
            record.seq().iter().copied().filter(|b| !gaps.contains(b)).collect()
        } else {
            record.seq().to_vec()
        };
        records.push((id, String::from_utf8(seq).map_err(|e| e.to_string())?));
    }

    Ok(records)
}

/// Writes an iterator of `AlignedSequence`s from Musals to a FASTA file at the given path.
#[expect(unused)]
pub fn write<P: AsRef<std::path::Path>>(path: &P, data: impl Iterator<Item = (String, String)>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut writer = fasta::Writer::to_file(path).map_err(|e| e.to_string())?;

    for (id, seq) in data {
        let record = fasta::Record::with_attrs(&id, None, seq.as_bytes());
        writer
            .write_record(&record)
            .map_err(|e| e.to_string())
            .map_err(|e| format!("Error while writing record: {e}, id: {id}, seq: {seq}"))?;
    }

    Ok(())
}
