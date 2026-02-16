//! Anomaly detection features applied as annotations for `Cluster`s and used for training meta-ML algorithms.

use std::collections::HashMap;

use crate::{Cluster, DistanceValue};

/// Anomaly detection features applied as annotations for `Cluster`s and used for training meta-ML algorithms.
#[derive(Debug, Clone, Copy, Default)]
pub struct Features {
    /// The ratio of the cardinality of the child to the parent cluster, accumulated along the path from the root to the cluster.
    pub cardinality_ratio: f64,
    /// The exponential moving average of cardinality ratios along the path from the root to the cluster.
    pub ema_cardinality_ratio: f64,
    /// The ratio of the radius of the child to the parent cluster.
    pub radius_ratio: f64,
    /// The exponential moving average of radius ratios along the path from the root to the cluster.
    pub ema_radius_ratio: f64,
    /// The ratio of the LFD of the child to the parent cluster.
    pub lfd_ratio: f64,
    /// The exponential moving average of LFD ratios along the path from the root to the cluster.
    pub ema_lfd_ratio: f64,
}

impl Features {
    /// For the root cluster, all ratios are 1.0 and all EMAs are 1.0 by definition.
    pub(crate) const fn for_root() -> Self {
        Self {
            cardinality_ratio: 1.0,
            ema_cardinality_ratio: 1.0,
            radius_ratio: 1.0,
            ema_radius_ratio: 1.0,
            lfd_ratio: 1.0,
            ema_lfd_ratio: 1.0,
        }
    }

    /// Given the parent cluster associated with these anomaly features, and a child cluster, compute the anomaly features for the child cluster.
    #[expect(clippy::cast_precision_loss)]
    pub(crate) fn for_child<T, A>(&self, parent: &Cluster<T, A>, child: &Cluster<T, A>) -> Self
    where
        T: DistanceValue,
    {
        let cardinality_ratio = (child.cardinality as f64 / parent.cardinality as f64) + self.cardinality_ratio;
        let radius_ratio = child
            .radius
            .to_f64()
            .and_then(|cr| parent.radius.to_f64().map(|pr| cr / pr))
            .unwrap_or_else(|| unreachable!("DistanceValue must be convertible to f64"));
        let lfd_ratio = child.lfd / parent.lfd;

        let ema_cardinality_ratio = next_ema(cardinality_ratio, self.ema_cardinality_ratio);
        let ema_radius_ratio = next_ema(radius_ratio, self.ema_radius_ratio);
        let ema_lfd_ratio = next_ema(lfd_ratio, self.ema_lfd_ratio);

        Self {
            cardinality_ratio,
            ema_cardinality_ratio,
            radius_ratio,
            ema_radius_ratio,
            lfd_ratio,
            ema_lfd_ratio,
        }
    }

    /// Applies gaussian error function normalization to all features.
    fn normalize(&mut self, mean: &Self, std_dev: &Self) {
        self.cardinality_ratio = libm::erf((self.cardinality_ratio - mean.cardinality_ratio) / std_dev.cardinality_ratio);
        self.ema_cardinality_ratio = libm::erf((self.ema_cardinality_ratio - mean.ema_cardinality_ratio) / std_dev.ema_cardinality_ratio);
        self.radius_ratio = libm::erf((self.radius_ratio - mean.radius_ratio) / std_dev.radius_ratio);
        self.ema_radius_ratio = libm::erf((self.ema_radius_ratio - mean.ema_radius_ratio) / std_dev.ema_radius_ratio);
        self.lfd_ratio = libm::erf((self.lfd_ratio - mean.lfd_ratio) / std_dev.lfd_ratio);
        self.ema_lfd_ratio = libm::erf((self.ema_lfd_ratio - mean.ema_lfd_ratio) / std_dev.ema_lfd_ratio);
    }
}

/// Applies gaussian error function normalization to all features in the collection.
#[expect(clippy::cast_precision_loss)]
pub fn normalize_features<H: core::hash::BuildHasher>(features: &mut HashMap<usize, Features, H>) {
    let count = features.len() as f64;
    let zeroed = Features {
        cardinality_ratio: 0.0,
        ema_cardinality_ratio: 0.0,
        radius_ratio: 0.0,
        ema_radius_ratio: 0.0,
        lfd_ratio: 0.0,
        ema_lfd_ratio: 0.0,
    };

    // Compute the mean for each feature.
    let mean = features.values().fold(zeroed, |acc, f| Features {
        cardinality_ratio: acc.cardinality_ratio + f.cardinality_ratio,
        ema_cardinality_ratio: acc.ema_cardinality_ratio + f.ema_cardinality_ratio,
        radius_ratio: acc.radius_ratio + f.radius_ratio,
        ema_radius_ratio: acc.ema_radius_ratio + f.ema_radius_ratio,
        lfd_ratio: acc.lfd_ratio + f.lfd_ratio,
        ema_lfd_ratio: acc.ema_lfd_ratio + f.ema_lfd_ratio,
    });
    let mean = Features {
        cardinality_ratio: mean.cardinality_ratio / count,
        ema_cardinality_ratio: mean.ema_cardinality_ratio / count,
        radius_ratio: mean.radius_ratio / count,
        ema_radius_ratio: mean.ema_radius_ratio / count,
        lfd_ratio: mean.lfd_ratio / count,
        ema_lfd_ratio: mean.ema_lfd_ratio / count,
    };

    // Compute the standard deviation for each feature.
    let std_dev = features.values().fold(zeroed, |acc, f| Features {
        cardinality_ratio: (f.cardinality_ratio - mean.cardinality_ratio).mul_add(f.cardinality_ratio - mean.cardinality_ratio, acc.cardinality_ratio),
        ema_cardinality_ratio: (f.ema_cardinality_ratio - mean.ema_cardinality_ratio)
            .mul_add(f.ema_cardinality_ratio - mean.ema_cardinality_ratio, acc.ema_cardinality_ratio),
        radius_ratio: (f.radius_ratio - mean.radius_ratio).mul_add(f.radius_ratio - mean.radius_ratio, acc.radius_ratio),
        ema_radius_ratio: (f.ema_radius_ratio - mean.ema_radius_ratio).mul_add(f.ema_radius_ratio - mean.ema_radius_ratio, acc.ema_radius_ratio),
        lfd_ratio: (f.lfd_ratio - mean.lfd_ratio).mul_add(f.lfd_ratio - mean.lfd_ratio, acc.lfd_ratio),
        ema_lfd_ratio: (f.ema_lfd_ratio - mean.ema_lfd_ratio).mul_add(f.ema_lfd_ratio - mean.ema_lfd_ratio, acc.ema_lfd_ratio),
    });
    let std_dev = Features {
        cardinality_ratio: (std_dev.cardinality_ratio / count).sqrt(),
        ema_cardinality_ratio: (std_dev.ema_cardinality_ratio / count).sqrt(),
        radius_ratio: (std_dev.radius_ratio / count).sqrt(),
        ema_radius_ratio: (std_dev.ema_radius_ratio / count).sqrt(),
        lfd_ratio: (std_dev.lfd_ratio / count).sqrt(),
        ema_lfd_ratio: (std_dev.ema_lfd_ratio / count).sqrt(),
    };

    // Normalize all features using the computed mean and standard deviation.
    for f in features.values_mut() {
        f.normalize(&mean, &std_dev);
    }
}

/// Compute the next exponential moving average of the given ratio and parent EMA.
///
/// The EMA is computed as `alpha * ratio + (1 - alpha) * parent_ema`, where `alpha`
/// is a constant value of `2 / 11`. This value was chosen because it gave the best
/// experimental results in the CHAODA paper.
///
/// # Arguments
///
/// * `ratio` - The ratio to compute the EMA of.
/// * `parent_ema` - The parent EMA to use.
#[must_use]
fn next_ema(ratio: f64, parent_ema: f64) -> f64 {
    // TODO: Consider getting `alpha` from user. Perhaps via env vars?
    let alpha = 2_f64 / 11.0;
    alpha.mul_add(ratio, (1_f64 - alpha) * parent_ema)
}
