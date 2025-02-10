//! The quality measures that may be calculated on a dimension reduction.

/// The quality measures that may be calculated on a dimension reduction.
#[derive(clap::ValueEnum, Debug, Clone, PartialEq, Eq)]
pub enum QualityMeasures {
    /// Calculate all available quality measures.
    #[clap(name = "all")]
    All,
    /// The distortion of a number of pair-wise distances.
    #[clap(name = "pairwise-distortion")]
    PairwiseDistortion,
    /// The distortion of a number of triangle inequalities, i.e. whether the
    /// edges of triangles still have the same relative lengths.
    #[clap(name = "triangle-inequality-distortion")]
    TriangleInequalityDistortion,
    /// The distortion of a number of angles between points.
    #[clap(name = "angle-distortion")]
    AngleDistortion,
}
