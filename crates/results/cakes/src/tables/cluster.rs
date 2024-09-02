//! The `Cluster` tables.

use std::{io::Write, path::Path, sync::Arc};

use abd_clam::Cluster;
use arrow::{
    array::{ArrayRef, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use serde::Serialize;

/// The fields of the `Cluster` table.
fn cluster_fields() -> Vec<Field> {
    vec![
        Field::new("depth", DataType::UInt32, false),
        Field::new("cardinality", DataType::UInt64, false),
        Field::new("radius", DataType::UInt32, false),
        Field::new("lfd", DataType::Float32, false),
        Field::new("arg_center", DataType::UInt64, false),
        Field::new("arg_radial", DataType::UInt64, false),
    ]
}

/// The schema of the `Ball` table.
pub fn ball_schema() -> Schema {
    Schema::new(cluster_fields())
}

/// The schema of the `SquishyBall` table.
pub fn squishy_ball_schema() -> Schema {
    let mut fields = cluster_fields();
    fields.push(Field::new("offset", DataType::UInt64, false));
    fields.push(Field::new("unitary_cost", DataType::UInt32, false));
    fields.push(Field::new("recursive_cost", DataType::UInt32, false));
    Schema::new(fields)
}

/// A single row of the `Ball` table.
#[derive(Serialize)]
pub struct BallRow {
    depth: u32,
    cardinality: u64,
    radius: u32,
    lfd: f32,
    arg_center: u64,
    arg_radial: u64,
}

impl BallRow {
    /// Convert a vector of `BallRow`s into a `RecordBatch`.
    pub fn into_record_batch(rows: Vec<Self>) -> RecordBatch {
        let (depth, cardinality, radius, lfd, arg_center, arg_radial) = rows.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            |(mut depth, mut cardinality, mut radius, mut lfd, mut arg_center, mut arg_radial), row| {
                depth.push(row.depth);
                cardinality.push(row.cardinality);
                radius.push(row.radius);
                lfd.push(row.lfd);
                arg_center.push(row.arg_center);
                arg_radial.push(row.arg_radial);
                (depth, cardinality, radius, lfd, arg_center, arg_radial)
            },
        );

        let depth = arrow::array::UInt32Array::from(depth);
        let cardinality = arrow::array::UInt64Array::from(cardinality);
        let radius = arrow::array::UInt32Array::from(radius);
        let lfd = arrow::array::Float32Array::from(lfd);
        let arg_center = arrow::array::UInt64Array::from(arg_center);
        let arg_radial = arrow::array::UInt64Array::from(arg_radial);

        let schema = ball_schema();

        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(depth) as ArrayRef,
                Arc::new(cardinality) as ArrayRef,
                Arc::new(radius) as ArrayRef,
                Arc::new(lfd) as ArrayRef,
                Arc::new(arg_center) as ArrayRef,
                Arc::new(arg_radial) as ArrayRef,
            ],
        )
        .unwrap_or_else(|e| unreachable!("{e}"))
    }

    /// Write a `Ball` to a CSV file.
    pub fn write_csv<P: AsRef<Path>>(ball: &crate::B, path: &P) -> Result<(), String> {
        let rows = ball
            .clone()
            .unstack_tree()
            .iter()
            .map(|(c, _, _)| Self::from(c))
            .map(|r| r.as_csv_row())
            .collect::<Vec<_>>();

        let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        file.write_all(Self::csv_header().as_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        for row in rows {
            file.write_all(row.as_bytes()).map_err(|e| e.to_string())?;
            file.write_all(b"\n").map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    fn csv_header() -> String {
        "depth,cardinality,radius,lfd,arg_center,arg_radial".to_string()
    }

    fn as_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.8},{},{}",
            self.depth, self.cardinality, self.radius, self.lfd, self.arg_center, self.arg_radial
        )
    }
}

impl From<&crate::B> for BallRow {
    #[allow(clippy::cast_possible_truncation)]
    fn from(ball: &crate::B) -> Self {
        Self {
            depth: ball.depth() as u32,
            cardinality: ball.cardinality() as u64,
            radius: ball.radius(),
            lfd: ball.lfd(),
            arg_center: ball.arg_center() as u64,
            arg_radial: ball.arg_radial() as u64,
        }
    }
}

/// A single row of the `SquishyBall` table.
#[derive(Serialize)]
pub struct SquishyBallRow {
    depth: u32,
    cardinality: u64,
    radius: u32,
    lfd: f32,
    arg_center: u64,
    arg_radial: u64,
    unitary_cost: u32,
    recursive_cost: u32,
    offset: u64,
}

impl SquishyBallRow {
    /// Convert a vector of `SquishyBallRow`s into a `RecordBatch`.
    pub fn into_record_batch(rows: Vec<Self>) -> RecordBatch {
        let (depth, cardinality, radius, lfd, arg_center, arg_radial, offset, unitary_cost, recursive_cost) =
            rows.into_iter().fold(
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
                |(
                    mut depth,
                    mut cardinality,
                    mut radius,
                    mut lfd,
                    mut arg_center,
                    mut arg_radial,
                    mut offset,
                    mut unitary_cost,
                    mut recursive_cost,
                ),
                 row| {
                    depth.push(row.depth);
                    cardinality.push(row.cardinality);
                    radius.push(row.radius);
                    lfd.push(row.lfd);
                    arg_center.push(row.arg_center);
                    arg_radial.push(row.arg_radial);
                    offset.push(row.offset);
                    unitary_cost.push(row.unitary_cost);
                    recursive_cost.push(row.recursive_cost);
                    (
                        depth,
                        cardinality,
                        radius,
                        lfd,
                        arg_center,
                        arg_radial,
                        offset,
                        unitary_cost,
                        recursive_cost,
                    )
                },
            );

        let depth = arrow::array::UInt32Array::from(depth);
        let cardinality = arrow::array::UInt64Array::from(cardinality);
        let radius = arrow::array::UInt32Array::from(radius);
        let lfd = arrow::array::Float32Array::from(lfd);
        let arg_center = arrow::array::UInt64Array::from(arg_center);
        let arg_radial = arrow::array::UInt64Array::from(arg_radial);
        let offset = arrow::array::UInt64Array::from(offset);
        let unitary_cost = arrow::array::UInt32Array::from(unitary_cost);
        let recursive_cost = arrow::array::UInt32Array::from(recursive_cost);

        let schema = squishy_ball_schema();

        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(depth) as ArrayRef,
                Arc::new(cardinality) as ArrayRef,
                Arc::new(radius) as ArrayRef,
                Arc::new(lfd) as ArrayRef,
                Arc::new(arg_center) as ArrayRef,
                Arc::new(arg_radial) as ArrayRef,
                Arc::new(offset) as ArrayRef,
                Arc::new(unitary_cost) as ArrayRef,
                Arc::new(recursive_cost) as ArrayRef,
            ],
        )
        .unwrap_or_else(|e| unreachable!("{e}"))
    }

    /// Write a `SquishyBall` to a CSV file.
    pub fn write_csv<P: AsRef<Path>>(ball: &crate::SB, path: &P) -> Result<(), String> {
        let rows = ball
            .clone()
            .unstack_tree()
            .iter()
            .map(|(c, _, _)| Self::from(c))
            .map(|r| r.as_csv_row())
            .collect::<Vec<_>>();

        let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        file.write_all(Self::csv_header().as_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        for row in rows {
            file.write_all(row.as_bytes()).map_err(|e| e.to_string())?;
            file.write_all(b"\n").map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    fn csv_header() -> String {
        "depth,cardinality,radius,lfd,arg_center,arg_radial,offset,unitary_cost,recursive_cost".to_string()
    }

    fn as_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.8},{},{},{},{},{}",
            self.depth,
            self.cardinality,
            self.radius,
            self.lfd,
            self.arg_center,
            self.arg_radial,
            self.offset,
            self.unitary_cost,
            self.recursive_cost
        )
    }
}

impl From<&crate::SB> for SquishyBallRow {
    #[allow(clippy::cast_possible_truncation)]
    fn from(ball: &crate::SB) -> Self {
        Self {
            depth: ball.depth() as u32,
            cardinality: ball.cardinality() as u64,
            radius: ball.radius(),
            lfd: ball.lfd(),
            arg_center: ball.arg_center() as u64,
            arg_radial: ball.arg_radial() as u64,
            unitary_cost: ball.unitary_cost(),
            recursive_cost: ball.recursive_cost(),
            offset: ball.offset() as u64,
        }
    }
}
