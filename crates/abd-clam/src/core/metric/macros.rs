//! Some macros for implementing `Metric` and `ParMetric` for smart pointers.

/// Implements `Metric` for a smart pointer.
macro_rules! impl_metric_block {
    () => {
        fn distance(&self, a: &I, b: &I) -> T {
            (**self).distance(a, b)
        }

        fn name(&self) -> &str {
            (**self).name()
        }

        fn has_identity(&self) -> bool {
            (**self).has_identity()
        }

        fn has_non_negativity(&self) -> bool {
            (**self).has_non_negativity()
        }

        fn has_symmetry(&self) -> bool {
            (**self).has_symmetry()
        }

        fn obeys_triangle_inequality(&self) -> bool {
            (**self).obeys_triangle_inequality()
        }

        fn is_expensive(&self) -> bool {
            (**self).is_expensive()
        }
    };
}

/// Implements `ParMetric` for a smart pointer.
macro_rules! impl_par_metric_block {
    () => {
        fn par_distance(&self, a: &I, b: &I) -> T {
            (**self).distance(a, b)
        }
    };
}

pub(crate) use impl_metric_block;
pub(crate) use impl_par_metric_block;
