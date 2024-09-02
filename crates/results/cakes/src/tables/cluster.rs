//! The `Cluster` tables.

use abd_clam::Cluster;
use arrow::datatypes::{DataType, Field, Schema};
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
