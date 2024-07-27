//! The `Children` of a `Cluster`.

use distances::Number;

/// The `Children` of a `Cluster`.
#[derive(Debug, Clone)]
pub struct Children<U: Number, C> {
    /// The children of the `Cluster`.
    children: Vec<Box<C>>,
    /// The indices of the poles used to partition the `Cluster`.
    arg_poles: Vec<usize>,
    /// The pairwise distances between the poles.
    polar_distances: Vec<Vec<U>>,
}

impl<U: Number, C> Children<U, C> {
    /// Creates a new `Children`.
    ///
    /// # Arguments
    ///
    /// - `children`: The children of the `Cluster`.
    /// - `arg_poles`: The indices of the poles used to partition the `Cluster`.
    pub fn new(children: Vec<C>, arg_poles: Vec<usize>, polar_distances: Vec<Vec<U>>) -> Self {
        Self {
            children: children.into_iter().map(Box::new).collect(),
            arg_poles,
            polar_distances,
        }
    }

    /// Decomposes the `Children` into its components.
    pub(crate) fn take(self) -> (Vec<C>, Vec<usize>, Vec<Vec<U>>) {
        let children = self.children.into_iter().map(|c| *c).collect();
        (children, self.arg_poles, self.polar_distances)
    }

    /// Returns the children of the `Cluster`.
    #[must_use]
    pub fn clusters(&self) -> Vec<&C> {
        self.children.iter().map(AsRef::as_ref).collect::<Vec<_>>()
    }

    /// Returns the children of the `Cluster` as mutable references.
    #[must_use]
    pub fn clusters_mut(&mut self) -> Vec<&mut C> {
        self.children.iter_mut().map(AsMut::as_mut).collect::<Vec<_>>()
    }

    /// Returns the indices of the poles used to partition the `Cluster`.
    #[must_use]
    pub fn arg_poles(&self) -> &[usize] {
        &self.arg_poles
    }

    /// Returns the pairwise distances between the poles.
    #[must_use]
    pub fn polar_distances(&self) -> &[Vec<U>] {
        &self.polar_distances
    }
}
