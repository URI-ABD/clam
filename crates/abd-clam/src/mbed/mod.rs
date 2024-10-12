//! Dimension Reduction with CLAM.

mod physics;

pub use physics::System as MassSpringSystem;

// TODO: Add ability to replace a cluster with its children in the mass-spring system.
// This would involve adding the child masses to the system as a triangle with the parent mass, letting the system
// evolve for a few steps, and then removing the parent mass.

// TODO: Embed every point from the dataset instead of just the cluster centers.
// This could be done by replacing parent masses with their children, as described above, until there are no parent
// masses remaining.

// TODO: Add methods for allowing only "local" evolution of the mass-spring system.
// This would require that we update only those masses which are within a small number spring connections of some target
// mass. This would allow for a more efficient implementation of the iterative process described above.

// TODO: Add methods for calculating the "accuracy" metrics of the dimensionality reduction.

// TODO: Break up the system into smaller components, using the `Graph` `Component`s.
// Each component would evolve independently (in parallel) until they all reach steady state. Then, the components would
// be made rigid, and we would combine them into a single system using some heuristic for adding springs between the
// components. Then the system would evolve as a whole until it reaches steady state.
