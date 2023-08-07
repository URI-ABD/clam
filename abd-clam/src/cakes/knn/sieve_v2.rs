//! Helper functions for Knn thresholds approach with separate grains for
//! cluster centers.

use core::{cmp::Ordering, marker::PhantomData};

use distances::Number;
use priority_queue::DoublePriorityQueue;

use crate::{Cluster, Dataset, Tree};
