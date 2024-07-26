//! A `Vertex` to use as a node in a `Graph`.

use distances::Number;

use crate::{Ball, Children};

/// A `Vertex` to use as a node in a `Graph`.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Vertex<U: Number, const N: usize> {
    /// The `Ball` that was adapted into this `Vertex`.
    ball: Ball<U>,
    /// The children of the `Vertex`.
    children: Option<Children<U, Self>>,
    /// The anomaly detection properties of the `Vertex`.
    ratios: [f32; N],
    /// The accumulated child-parent cardinality ratio.
    accumulated_cp_car_ratio: f32,
}
