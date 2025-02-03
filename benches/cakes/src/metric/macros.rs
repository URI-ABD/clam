//! Some macros for implementing `CountingMetric` for smart pointers.

/// Implements `CountingMetric` for a smart pointer.
#[macro_export]
macro_rules! impl_counting_metric_for_smart_pointer {
    () => {
        fn disable_counting(&mut self) {
            self.as_mut().disable_counting();
        }

        fn enable_counting(&mut self) {
            self.as_mut().enable_counting();
        }

        fn count(&self) -> usize {
            self.as_ref().count()
        }

        fn reset_count(&self) -> usize {
            self.as_ref().reset_count()
        }

        fn increment(&self) {
            self.as_ref().increment();
        }
    };
}
