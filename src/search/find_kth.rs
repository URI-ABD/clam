use std::cmp::Ordering;

use super::knn_sieve;

use crate::prelude::*;

pub fn find_kth_d<'a, T: Number>(
    grains: &mut [knn_sieve::Grain<'a, T>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
) -> (knn_sieve::Grain<'a, T>, usize) {
    assert_eq!(grains.len(), cumulative_cardinalities.len());
    _find_kth(
        grains,
        cumulative_cardinalities,
        k,
        0,
        grains.len() - 1,
        &knn_sieve::Delta::D,
    )
}

pub fn find_kth_d_max<'a, T: Number>(
    grains: &mut [knn_sieve::Grain<'a, T>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
) -> (knn_sieve::Grain<'a, T>, usize) {
    assert_eq!(grains.len(), cumulative_cardinalities.len());
    _find_kth(
        grains,
        cumulative_cardinalities,
        k,
        0,
        grains.len() - 1,
        &knn_sieve::Delta::Max,
    )
}

pub fn find_kth_d_min<'a, T: Number>(
    grains: &mut [knn_sieve::Grain<'a, T>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
) -> (knn_sieve::Grain<'a, T>, usize) {
    assert_eq!(grains.len(), cumulative_cardinalities.len());
    _find_kth(
        grains,
        cumulative_cardinalities,
        k,
        0,
        grains.len() - 1,
        &knn_sieve::Delta::Min,
    )
}

pub fn _find_kth<'a, T: Number>(
    grains: &mut [knn_sieve::Grain<'a, T>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
    l: usize,
    r: usize,
    delta: &knn_sieve::Delta,
) -> (knn_sieve::Grain<'a, T>, usize) {
    let mut cardinalities = (0..1)
        .chain(cumulative_cardinalities.iter().copied())
        .zip(cumulative_cardinalities.iter().copied())
        .map(|(prev, next)| next - prev)
        .collect::<Vec<_>>();
    assert_eq!(cardinalities.len(), grains.len());

    let position = partition(grains, &mut cardinalities, l, r, delta);

    let mut cumulative_cardinalities = cardinalities
        .iter()
        .scan(0, |acc, v| {
            *acc += *v;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    match cumulative_cardinalities[position].cmp(&k) {
        Ordering::Less => _find_kth(grains, &mut cumulative_cardinalities, k, position + 1, r, delta),
        Ordering::Equal => (grains[position].clone(), position),
        Ordering::Greater => {
            if (position > 0) && (cumulative_cardinalities[position - 1] > k) {
                _find_kth(grains, &mut cumulative_cardinalities, k, l, position - 1, delta)
            } else {
                (grains[position].clone(), position)
            }
        }
    }
}

pub fn find_kth_threshold<U: Number>(thresholds: &mut [(U, usize)], k: usize) -> (usize, (U, usize)) {
    let (index, (threshold, _)) = _find_kth_threshold(thresholds, k, 0, thresholds.len() - 1);

    let mut b = index;
    for a in (index + 1)..(thresholds.len()) {
        if thresholds[a].0 == threshold {
            b += 1;
            thresholds.swap(a, b);
        }
    }

    (b, thresholds[b])
}

fn _find_kth_threshold<U: Number>(thresholds: &mut [(U, usize)], k: usize, l: usize, r: usize) -> (usize, (U, usize)) {
    if l >= r {
        let position = std::cmp::min(l, r);
        (position, thresholds[position])
    } else {
        let position = partition_threshold(thresholds, l, r);
        let guaranteed_cardinalities = thresholds
            .iter()
            .scan(0, |acc, &(_, v)| {
                *acc += v;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        assert!(
            guaranteed_cardinalities[r] > k,
            "Too few guarantees {} vs {} at {} ...",
            guaranteed_cardinalities[r],
            k,
            r
        );

        let num_guaranteed = guaranteed_cardinalities[position];

        match num_guaranteed.cmp(&k) {
            Ordering::Less => _find_kth_threshold(thresholds, k, position + 1, r),
            Ordering::Equal => (position, thresholds[position]),
            Ordering::Greater => {
                if (position > 0) && (guaranteed_cardinalities[position - 1] > k) {
                    _find_kth_threshold(thresholds, k, l, position - 1)
                } else {
                    (position, thresholds[position])
                }
            }
        }
    }
}

pub fn partition_threshold<U: Number>(thresholds: &mut [(U, usize)], l: usize, r: usize) -> usize {
    let pivot = (l + r) / 2; // Check for overflow
    thresholds.swap(pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        if thresholds[b].0 <= thresholds[r].0 {
            thresholds.swap(a, b);
            a += 1;
        }
        b += 1;
    }

    thresholds.swap(a, r);

    a
}

fn partition<T: Number>(
    grains: &mut [knn_sieve::Grain<T>],
    cardinalities: &mut [usize],
    l: usize,
    r: usize,
    delta: &knn_sieve::Delta,
) -> usize {
    let pivot = (l + r) / 2; // Check for overflow
    swaps(grains, cardinalities, pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        let grain_comp = match delta {
            knn_sieve::Delta::D => (grains[b]).ord_by_d(&grains[r]),
            knn_sieve::Delta::Max => (grains[b]).ord_by_d_max(&grains[r]),
            knn_sieve::Delta::Min => (grains[b]).ord_by_d_min(&grains[r]),
        };

        if grain_comp == Ordering::Less {
            swaps(grains, cardinalities, a, b);
            a += 1;
        }
        b += 1;
    }

    swaps(grains, cardinalities, a, r);

    a
}

pub fn swaps<T>(grains: &mut [T], cardinalities: &mut [usize], i: usize, j: usize) {
    grains.swap(i, j);
    cardinalities.swap(i, j);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn find_kth_vs_sort() {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..1000)
            .map(|_| (0..5).map(|_| rng.gen_range(0.0..10.0)).collect())
            .collect();
        let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
        let metric = metric_from_name::<f64>("euclideansq", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref());

        let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = crate::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);
        let flat_tree = cluster.subtree();

        let query_ind = rng.gen_range(0..data.len());
        let k = 42;

        let mut sieve = knn_sieve::KnnSieve::new(flat_tree.clone(), &data[query_ind], k);

        let kth_grain = find_kth_d_max(&mut sieve.grains, &mut sieve.cumulative_cardinalities, sieve.k);

        sieve.grains.sort_by(|a, b| a.ord_by_d_max(b));
        sieve.update_cumulative_cardinalities();
        sieve.update_guaranteed_cardinalities();
        let index = sieve
            .cumulative_cardinalities
            .iter()
            .position(|c| *c >= sieve.k)
            .unwrap();

        let threshold_from_sort = sieve.grains[index].d_max;
        assert_eq!(kth_grain.0.d_max, threshold_from_sort);
    }

    #[test]
    fn kth_relative_position() {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..1000)
            .map(|_| (0..5).map(|_| rng.gen_range(0.0..10.0)).collect())
            .collect();
        let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
        let metric = metric_from_name::<f64>("euclideansq", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref());

        let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = crate::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);
        let flat_tree = cluster.subtree();

        let query_ind = rng.gen_range(0..data.len());
        let k = 42;

        let mut sieve = knn_sieve::KnnSieve::new(flat_tree.clone(), &data[query_ind], k);

        let kth_grain = find_kth_d_min(&mut sieve.grains, &mut sieve.cumulative_cardinalities, sieve.k);

        for i in 0..sieve.grains.clone().len() {
            if i < kth_grain.1 {
                assert!(sieve.grains[i].d_min <= kth_grain.0.d_min);
            } else {
                assert!(sieve.grains[i].d_min >= kth_grain.0.d_min);
            }
        }
    }
}
