//! Dimension Reduction with CLAM.

mod mss;
mod spring;
mod system;
mod vector;

pub use mss::System as MSS;
pub use spring::Spring;
pub use system::System as MassSpringSystem;
pub use vector::Vector;

// TODO: Add methods for allowing only "local" evolution of the mass-spring system.
// This would require that we update only those masses which are within a small number spring connections of some target
// mass. This would allow us to simulate the system in a more efficient manner.

// TODO: Add methods for calculating the "accuracy" metrics of the dimensionality reduction.
