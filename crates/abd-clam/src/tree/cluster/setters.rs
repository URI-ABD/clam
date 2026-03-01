//! Setters for `Cluster` properties.

use super::Cluster;

/// Setters for `Cluster` properties.
impl<T, A> Cluster<T, A> {
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

    /// Updates the annotation of this cluster, returning the old annotation.
    pub const fn update_annotation(&mut self, annotation: A) -> A {
        core::mem::replace(&mut self.annotation, annotation)
    }

    /// Updates the annotation of this cluster using the provided function, returning the old annotation.
    ///
    /// The provided function is called before the annotation is updated, so it can use the current annotation if needed.
    pub fn update_annotation_with<F>(&mut self, f: F) -> A
    where
        F: FnOnce(&Self) -> A,
    {
        self.update_annotation(f(self))
    }

    /// Changes the type of the annotation of this cluster, returning the old annotation.
    pub fn change_annotation<B>(self, annotation: B) -> (Cluster<T, B>, A) {
        let old_annotation = self.annotation;
        let cluster = Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children: self.children,
            annotation,
            parent_center_index: self.parent_center_index,
        };
        (cluster, old_annotation)
    }

    /// Changes the type of the annotation of this cluster using the provided function, returning the old annotation.
    ///
    /// The provided function is called before the annotation is updated, so it can use the current annotation if needed.
    pub fn change_annotation_with<F, B>(self, f: F) -> (Cluster<T, B>, A)
    where
        F: FnOnce(&Self) -> B,
    {
        let new_annotation = f(&self);
        self.change_annotation(new_annotation)
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
