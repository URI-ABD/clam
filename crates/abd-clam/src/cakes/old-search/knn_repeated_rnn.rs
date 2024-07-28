//! K-Nearest Neighbors search using repeated Clustered RNN search.

use distances::{number::Multiplication, Number};

use crate::core::{cluster::LFD, Cluster, Dataset, ParDataset, SizedHeap};

use super::rnn_clustered::{leaf_search, par_leaf_search, par_tree_search, tree_search};

/// K-Nearest Neighbors search using repeated Clustered RNN search.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize, max_multiplier: U) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let max_multiplier = max_multiplier.as_f32();
    let mut radius = root.radius().as_f32();

    let mut multiplier = LFD::multiplier_for_k(root.lfd(), root.cardinality(), k).min(max_multiplier);
    radius = radius.mul_add(multiplier, U::EPSILON.as_f32());

    let [mut confirmed, mut straddlers] = tree_search(data, root, query, U::from(radius));

    let mut num_confirmed = count_hits(&confirmed);
    while num_confirmed == 0 {
        radius = radius.double();
        [confirmed, straddlers] = tree_search(data, root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    while num_confirmed < k {
        let (lfd, car) = mean_lfd(&confirmed, &straddlers);
        multiplier = LFD::multiplier_for_k(lfd, car, k)
            .min(max_multiplier)
            .max(f32::ONE + f32::EPSILON);
        radius = radius.mul_add(multiplier, U::EPSILON.as_f32());
        [confirmed, straddlers] = tree_search(data, root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    let mut knn = SizedHeap::new(Some(k));
    leaf_search(data, confirmed, straddlers, query, U::from(radius))
        .into_iter()
        .for_each(|(i, d)| knn.push((d, i)));
    knn.items().map(|(d, i)| (i, d)).collect()
}

/// Parallel K-Nearest Neighbors search using repeated Clustered RNN search.
pub fn par_search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize, max_multiplier: U) -> Vec<(usize, U)>
where
    I: Send + Sync,
    U: Number,
    D: ParDataset<I, U>,
    C: Cluster<U> + Send + Sync,
{
    let max_multiplier = max_multiplier.as_f32();
    let mut radius = root.radius().as_f32();

    let mut multiplier = LFD::multiplier_for_k(root.lfd(), root.cardinality(), k).min(max_multiplier);
    radius = radius.mul_add(multiplier, U::EPSILON.as_f32());

    let [mut confirmed, mut straddlers] = par_tree_search(data, root, query, U::from(radius));

    let mut num_confirmed = count_hits(&confirmed);
    while num_confirmed == 0 {
        radius = radius.mul_add(max_multiplier, U::EPSILON.as_f32());
        [confirmed, straddlers] = par_tree_search(data, root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    while num_confirmed < k {
        let (lfd, car) = mean_lfd(&confirmed, &straddlers);
        multiplier = LFD::multiplier_for_k(lfd, car, k).min(max_multiplier);
        radius = radius.mul_add(multiplier, U::EPSILON.as_f32());
        [confirmed, straddlers] = par_tree_search(data, root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    let mut knn = SizedHeap::new(Some(k));
    par_leaf_search(data, confirmed, straddlers, query, U::from(radius))
        .into_iter()
        .for_each(|(i, d)| knn.push((d, i)));
    knn.items().map(|(d, i)| (i, d)).collect()
}

/// Count the total cardinality of the clusters.
fn count_hits<U: Number, C: Cluster<U>>(hits: &[(&C, U)]) -> usize {
    hits.iter().map(|(c, _)| c.cardinality()).sum()
}

/// Calculate the weighted mean of the LFDs of the clusters.
fn mean_lfd<U: Number, C: Cluster<U>>(confirmed: &[(&C, U)], straddlers: &[(&C, U)]) -> (f32, usize) {
    let (lfd, car) = confirmed
        .iter()
        .map(|&(c, _)| c)
        .chain(straddlers.iter().map(|&(c, _)| c))
        .map(|c| (c.lfd(), c.cardinality()))
        .fold((0.0, 0), |(lfd, car), (l, c)| (l.mul_add(c.as_f32(), lfd), car + c));
    (lfd / car.as_f32(), car)
}

#[cfg(test)]
mod tests {
    use crate::{
        cakes::search::tests::gen_grid_data,
        core::{
            cluster::{Ball, Partition},
            LinearSearch, ParLinearSearch,
        },
    };

    use super::super::tests::{check_search_by_distance, gen_line_data};

    use super::*;

    #[test]
    fn line() -> Result<(), String> {
        let mut data = gen_line_data(10)?;
        let query = &0;

        let criteria = |c: &Ball<u32>| c.cardinality() > 1;
        let seed = Some(42);
        let (root, indices) = Ball::new_tree_and_permute(&mut data, &criteria, seed);

        println!();
        println!("Re-joined Indices: {indices:?}");
        println!(
            "Instances: {:?}",
            (0..data.cardinality())
                .zip(data.instances.iter())
                .zip(data.metadata.iter())
                .map(|((i, x), m)| (i, x, m))
                .collect::<Vec<_>>()
        );

        let max_multiplier = 2;

        for k in [1, 4, 8] {
            let true_hits = data.knn(query, k);
            assert_eq!(true_hits.len(), k, "Linear search failed: {true_hits:?}");

            let pred_hits = search(&data, &root, query, k, max_multiplier);
            assert_eq!(pred_hits.len(), k, "KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(true_hits.clone(), pred_hits, "KnnRepeatedRnn"));

            let pred_hits = data.par_knn(query, k);
            assert_eq!(pred_hits.len(), k, "Parallel linear search failed: {pred_hits:?}");
            assert!(check_search_by_distance(true_hits.clone(), pred_hits, "KnnParLinear"));

            let pred_hits = par_search(&data, &root, query, k, max_multiplier);
            assert_eq!(pred_hits.len(), k, "Parallel KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(true_hits, pred_hits, "ParKnnRepeatedRnn"));
        }

        Ok(())
    }

    #[test]
    fn grid() -> Result<(), String> {
        let mut data = gen_grid_data(10)?;
        let query = &(0.0, 0.0);

        let criteria = |c: &Ball<f32>| c.cardinality() > 1;
        let seed = Some(42);
        let (root, _) = Ball::new_tree_and_permute(&mut data, &criteria, seed);
        let max_multiplier = 2.0;

        for k in [1, 4, 8, 16, 32] {
            let true_hits = data.knn(query, k);
            assert_eq!(true_hits.len(), k, "Linear search failed: {true_hits:?}");

            let pred_hits = search(&data, &root, query, k, max_multiplier);
            assert_eq!(pred_hits.len(), k, "KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(
                true_hits.clone(),
                pred_hits,
                &format!("KnnRepeatedRnn k={k}")
            ));

            let pred_hits = par_search(&data, &root, query, k, max_multiplier);
            assert_eq!(pred_hits.len(), k, "Parallel KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(
                true_hits,
                pred_hits,
                &format!("Par KnnRepeatedRnn k={k}")
            ));
        }

        Ok(())
    }
}
