//! A `Metric` is a wrapper for a distance function that provides information
//! about the properties of the distance function.

use distances::Number;

use super::{MetricSpace, ParMetricSpace};

/// A `Metric` is a wrapper for a distance function that provides information
/// about the properties of the distance function.
///
/// # Type Parameters
///
/// - `I`: The type of inputs to the distance function.
/// - `U`: The type of the distance values.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
pub struct Metric<I, U> {
    /// Whether the distance function provides an identity.
    pub(crate) identity: bool,
    /// Whether the distance function is non-negative.
    pub(crate) non_negativity: bool,
    /// Whether the distance function is symmetric.
    pub(crate) symmetry: bool,
    /// Whether the distance function satisfies the triangle inequality.
    pub(crate) triangle_inequality: bool,
    /// Whether the distance function is expensive to compute.
    pub(crate) expensive: bool,
    /// The distance function.
    pub(crate) distance_function: fn(&I, &I) -> U,
}

impl<I, U> Metric<I, U> {
    /// Creates a new `Metric`.
    ///
    /// This sets the `identity`, `non_negativity`, `symmetry`, and
    /// `triangle_inequality` properties to `true`.
    ///
    /// # Parameters
    ///
    /// - `distance_function`: The distance function.
    /// - `expensive`: Whether the distance function is expensive to compute.
    pub fn new(distance_function: fn(&I, &I) -> U, expensive: bool) -> Self {
        Self {
            identity: true,
            non_negativity: true,
            symmetry: true,
            triangle_inequality: true,
            expensive,
            distance_function,
        }
    }

    /// Specifies that this distance function provides an identity.
    #[must_use]
    pub const fn has_identity(mut self) -> Self {
        self.identity = true;
        self
    }

    /// Specifies that this distance function does not provide an identity.
    #[must_use]
    pub const fn no_identity(mut self) -> Self {
        self.identity = false;
        self
    }

    /// Specifies that this distance function is non-negative.
    #[must_use]
    pub const fn has_non_negativity(mut self) -> Self {
        self.non_negativity = true;
        self
    }

    /// Specifies that this distance function is not non-negative.
    #[must_use]
    pub const fn no_non_negativity(mut self) -> Self {
        self.non_negativity = false;
        self
    }

    /// Specifies that this distance function is symmetric.
    #[must_use]
    pub const fn has_symmetry(mut self) -> Self {
        self.symmetry = true;
        self
    }

    /// Specifies that this distance function is not symmetric.
    #[must_use]
    pub const fn no_symmetry(mut self) -> Self {
        self.symmetry = false;
        self
    }

    /// Specifies that this distance function satisfies the triangle inequality.
    #[must_use]
    pub const fn has_triangle_inequality(mut self) -> Self {
        self.triangle_inequality = true;
        self
    }

    /// Specifies that this distance function does not satisfy the triangle
    /// inequality.
    #[must_use]
    pub const fn no_triangle_inequality(mut self) -> Self {
        self.triangle_inequality = false;
        self
    }

    /// Specifies that this distance function is expensive to compute.
    #[must_use]
    pub const fn is_expensive(mut self) -> Self {
        self.expensive = true;
        self
    }

    /// Specifies that this distance function is not expensive to compute.
    #[must_use]
    pub const fn is_not_expensive(mut self) -> Self {
        self.expensive = false;
        self
    }
}

impl<I, U: Number> MetricSpace<I, U> for Metric<I, U> {
    fn metric(&self) -> &Self {
        self
    }

    fn identity(&self) -> bool {
        self.identity
    }

    fn non_negativity(&self) -> bool {
        self.non_negativity
    }

    fn symmetry(&self) -> bool {
        self.symmetry
    }

    fn triangle_inequality(&self) -> bool {
        self.triangle_inequality
    }

    fn expensive(&self) -> bool {
        self.expensive
    }

    fn distance_function(&self) -> fn(&I, &I) -> U {
        self.distance_function
    }
}

impl<I: Send + Sync, U: Number> ParMetricSpace<I, U> for Metric<I, U> {}
