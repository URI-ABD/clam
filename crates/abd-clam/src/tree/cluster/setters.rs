//! Setters for `Cluster` properties.

use super::Cluster;

/// Setters for `Cluster` properties.
impl<T, A> Cluster<T, A> {
    /// Increments the index of the center item and the indices of all child centers by the given offset.
    pub(crate) fn increment_indices(&mut self, offset: usize) {
        self.center_index += offset;
        if let Some((child_center_indices, _)) = &mut self.children {
            for ci in child_center_indices.iter_mut() {
                *ci += offset;
            }
        }
    }

    /// Compounds the annotation of this cluster with the additional annotation.
    pub fn compound_annotation<B>(self, additional_annotation: B) -> Cluster<T, (A, B)> {
        Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children: self.children,
            annotation: (self.annotation, additional_annotation),
            parent_center_index: self.parent_center_index,
        }
    }

    /// Sets the annotation of this cluster, replacing any existing annotation.
    pub fn set_annotation(&mut self, annotation: A) {
        self.annotation = annotation;
    }
}

impl<T, A, B> Cluster<T, (A, B)> {
    /// Removes the additional annotation from this cluster, returning a cluster with the original annotation and the additional annotation as a separate value.
    pub fn decompound_annotation(self) -> (Cluster<T, A>, B) {
        let (original_annotation, additional_annotation) = self.annotation;
        let cluster = Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children: self.children,
            annotation: original_annotation,
            parent_center_index: self.parent_center_index,
        };
        (cluster, additional_annotation)
    }
}
