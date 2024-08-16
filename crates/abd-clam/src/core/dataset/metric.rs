//! A `Metric` is a wrapper for a distance function that provides information
//! about the properties of the distance function.

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
    /// The name of the distance function.
    pub(crate) name: String,
}

impl<I, U> Default for Metric<I, U> {
    fn default() -> Self {
        Self {
            identity: true,
            non_negativity: true,
            symmetry: true,
            triangle_inequality: true,
            expensive: false,
            distance_function: |_, _| unreachable!("This should never be called."),
            name: "Unknown Metric".to_string(),
        }
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl<I, U> core::fmt::Debug for Metric<I, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metric")
            .field("name", &self.name)
            .field("identity", &self.identity)
            .field("non_negativity", &self.non_negativity)
            .field("symmetry", &self.symmetry)
            .field("triangle_inequality", &self.triangle_inequality)
            .field("expensive", &self.expensive)
            .finish()
    }
}

impl<I, U> core::fmt::Display for Metric<I, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} Metric", self.name)
    }
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
            name: "Unknown Metric".to_string(),
        }
    }

    /// Returns the name of the distance function.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets the name of the distance function.
    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
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
