//! Generating tables of data from `Cluster`s to plot later.

use std::{path::Path, sync::Arc};

use abd_clam::Cluster;
use arrow::ipc::writer::FileWriter;
use arrow_json::ReaderBuilder;

mod cluster;

/// Generate a table from a `Ball`.
pub fn write_ball_table<P: AsRef<Path>>(ball: &crate::B, path: &P) -> Result<(), String> {
    let schema = cluster::ball_schema();
    let rows = ball
        .clone()
        .unstack_tree()
        .iter()
        .map(|(c, _, _)| cluster::BallRow::from(c))
        .collect::<Vec<_>>();
    let mut decoder = ReaderBuilder::new(Arc::new(schema))
        .build_decoder()
        .unwrap_or_else(|e| unreachable!("{e}"));
    decoder.serialize(&rows).unwrap_or_else(|e| unreachable!("{e}"));

    let table = decoder
        .flush()
        .unwrap_or_else(|e| unreachable!("{e}"))
        .unwrap_or_else(|| unreachable!("No table"));

    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = FileWriter::try_new(&mut file, table.schema().as_ref()).map_err(|e| e.to_string())?;
    writer.write(&table).map_err(|e| e.to_string())?;

    Ok(())
}

/// Generate a table from a `SquishyBall`.
pub fn write_squishy_ball_table<P: AsRef<Path>>(ball: &crate::SB, path: &P) -> Result<(), String> {
    let schema = cluster::squishy_ball_schema();
    let rows = ball
        .clone()
        .unstack_tree()
        .iter()
        .map(|(c, _, _)| cluster::SquishyBallRow::from(c))
        .collect::<Vec<_>>();
    let mut decoder = ReaderBuilder::new(Arc::new(schema))
        .build_decoder()
        .unwrap_or_else(|e| unreachable!("{e}"));
    decoder.serialize(&rows).unwrap_or_else(|e| unreachable!("{e}"));

    let table = decoder
        .flush()
        .unwrap_or_else(|e| unreachable!("{e}"))
        .unwrap_or_else(|| unreachable!("No table"));

    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = FileWriter::try_new(&mut file, table.schema().as_ref()).map_err(|e| e.to_string())?;
    writer.write(&table).map_err(|e| e.to_string())?;

    Ok(())
}
