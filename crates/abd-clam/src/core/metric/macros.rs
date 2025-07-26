//! Some macros for implementing `Metric` and `ParMetric` for smart pointers.

/// Implementation block for `Metric`.
#[macro_export]
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

/// Implementation block for `ParMetric`.
#[macro_export]
macro_rules! impl_par_metric_block {
    () => {
        fn par_distance(&self, a: &I, b: &I) -> T {
            (**self).distance(a, b)
        }
    };
}

/// Implements `Metric` for several smart pointers.
#[macro_export]
macro_rules! impl_metric_for_smart_pointers {
    ($($pointer:tt),*) => {
        $(
            impl<I, T: Number> Metric<I, T> for $pointer<dyn Metric<I, T>> {
                impl_metric_block!();
            }

            impl<I: Send + Sync, T: Number> Metric<I, T> for $pointer<dyn ParMetric<I, T>> {
                impl_metric_block!();
            }
        )*
    };
}

/// Implements `ParMetric` for several smart pointers.
#[macro_export]
macro_rules! impl_par_metric_for_smart_pointers {
    ($($pointer:tt),*) => {
        $(
            impl<I: Send + Sync, T: Number> ParMetric<I, T> for $pointer<dyn ParMetric<I, T>> {
                impl_par_metric_block!();
            }
        )*
    };
}
