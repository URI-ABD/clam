use std::path::Path;

use abd_clam::musals::Sequence;
use bio::io::fasta;

/// A wrapper type on String to allow use of the `Sequence` trait.
#[derive(Debug, Clone, PartialEq, Eq, Default, databuf::Encode, databuf::Decode, serde::Deserialize, serde::Serialize)]
pub struct MusalsSequence(pub String);

impl core::fmt::Display for MusalsSequence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<[u8]> for MusalsSequence {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl Sequence for MusalsSequence {
    const GAP: u8 = b'-';

    #[allow(clippy::expect_used)]
    fn from_vec(v: Vec<u8>) -> Self {
        Self(String::from_utf8(v).expect("Invalid UTF-8 sequence"))
    }

    fn from_string(s: String) -> Self {
        Self(s)
    }

    fn append(&mut self, other: Self) {
        self.0.push_str(&other.0);
    }

    fn pre_pend(&mut self, byte: u8) {
        self.0.insert(0, byte as char);
    }

    fn post_pend(&mut self, byte: u8) {
        self.0.push(byte as char);
    }
}

/// Reads a FASTA file from the given path.
pub fn read<P: AsRef<Path>>(path: P, remove_gaps: bool) -> Result<Vec<(String, MusalsSequence)>, String> {
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
        records.push((id, MusalsSequence::from_vec(seq)));
    }

    Ok(records)
}

/// Writes a FASTA file to the given path.
pub fn write<P: AsRef<Path>>(path: P, data: &[(String, MusalsSequence)]) -> Result<(), String> {
    let mut writer = fasta::Writer::to_file(&path).map_err(|e| e.to_string())?;

    for (id, seq) in data {
        let record = fasta::Record::with_attrs(id, None, seq.as_ref());
        writer
            .write_record(&record)
            .map_err(|e| e.to_string())
            .map_err(|e| format!("Error while writing record: {e}, id: {id}, seq: {seq}"))?;
    }

    Ok(())
}
