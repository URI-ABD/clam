//! Generating tables of data from `Cluster`s to plot later.

#![allow(dead_code)]

use std::path::Path;

use abd_clam::Cluster;
use arrow::ipc::writer::FileWriter;

mod cluster;

/// Write an arrow array of `Ball`s to the path
pub fn write_ball_table<P: AsRef<Path>>(ball: &crate::BSet, path: &P) -> Result<(), String> {
    let rows = ball
        .clone()
        .unstack_tree()
        .iter()
        .map(|(c, _, _)| cluster::BallRow::from(c))
        .collect::<Vec<_>>();

    let batch = cluster::BallRow::into_record_batch(rows);
    let schema = batch.schema();

    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = FileWriter::try_new(&mut file, &schema).map_err(|e| e.to_string())?;
    writer.write(&batch).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;

    Ok(())
}

/// Write a CSV file of `Ball`s to the path
pub fn write_ball_csv<P: AsRef<Path>>(ball: &crate::BSet, path: &P) -> Result<(), String> {
    cluster::BallRow::write_csv(ball, path)
}

/// Write an arrow array of `SquishyBall`s to the path
pub fn write_squishy_ball_table<P: AsRef<Path>>(ball: &crate::SBSet, path: &P) -> Result<(), String> {
    let rows = ball
        .clone()
        .unstack_tree()
        .iter()
        .map(|(c, _, _)| cluster::SquishyBallRow::from(c))
        .collect::<Vec<_>>();

    let batch = cluster::SquishyBallRow::into_record_batch(rows);
    let schema = batch.schema();

    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = FileWriter::try_new(&mut file, &schema).map_err(|e| e.to_string())?;
    writer.write(&batch).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;

    Ok(())
}

/// Write a CSV file of `SquishyBall`s to the path
pub fn write_squishy_ball_csv<P: AsRef<Path>>(ball: &crate::SBSet, path: &P) -> Result<(), String> {
    cluster::SquishyBallRow::write_csv(ball, path)
}
