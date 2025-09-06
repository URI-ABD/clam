//! Wrapper structs for implementing `Ord` on `PartialOrd` types for use within
//! CLAM.

/// A wrapper struct for implementing `Ord` on a `PartialOrd` in a way that is
/// suitable for use within CLAM.
///
/// A `MaxItem<A, T>` is ordered in the same way as `T`, except that
/// incomparable items are treated as less than any other item. The additional
/// type parameter `A` is used to store arbitrary data alongside the item and is
/// ignored for ordering purposes.
#[derive(Debug)]
pub struct MaxItem<A, T: PartialOrd>(pub A, pub T);

impl<A, T: PartialOrd> PartialEq for MaxItem<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<A, T: PartialOrd> Eq for MaxItem<A, T> {}

impl<A, T: PartialOrd> PartialOrd for MaxItem<A, T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<A, T: PartialOrd> Ord for MaxItem<A, T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.1.partial_cmp(&other.1).unwrap_or(core::cmp::Ordering::Less)
    }
}

/// A wrapper struct for implementing `Ord` on a `PartialOrd` in a way that is
/// suitable for use within CLAM.
///
/// A `MinItem<A, T>` is ordered in the same way as `T`, except that
/// incomparable items are treated as greater than any other item. The
/// additional type parameter `A` is used to store arbitrary data alongside the
/// item and is ignored for ordering purposes.
#[derive(Debug)]
pub struct MinItem<A, T: PartialOrd>(pub A, pub T);

impl<A, T: PartialOrd> PartialEq for MinItem<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<A, T: PartialOrd> Eq for MinItem<A, T> {}

impl<A, T: PartialOrd> PartialOrd for MinItem<A, T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<A, T: PartialOrd> Ord for MinItem<A, T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.1.partial_cmp(&other.1).unwrap_or(core::cmp::Ordering::Greater)
    }
}
