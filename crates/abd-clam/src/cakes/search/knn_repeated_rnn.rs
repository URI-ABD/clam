//! K-Nearest Neighbors search using repeated Clustered RNN search.

use distances::{number::Multiplication, Number};

use crate::{cluster::ParCluster, dataset::ParDataset, linear_search::SizedHeap, Cluster, Dataset, LFD};

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
    C: ParCluster<U>,
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
    use distances::Number;

    use crate::{
        cakes::OffsetBall,
        cluster::{Ball, ParCluster, Partition},
        linear_search::LinearSearch,
        Cluster, FlatVec,
    };

    use super::super::tests::{check_search_by_distance, gen_grid_data, gen_line_data};

    fn check_knn<I: Send + Sync, U: Number, C: ParCluster<U>>(
        root: &C,
        data: &FlatVec<I, U, usize>,
        query: &I,
        k: usize,
        max_multiplier: U,
    ) -> bool {
        let true_hits = data.knn(query, k);

        let pred_hits = super::search(data, root, query, k, max_multiplier);
        assert_eq!(pred_hits.len(), true_hits.len(), "Knn search failed: {pred_hits:?}");
        check_search_by_distance(true_hits.clone(), pred_hits, "KnnClustered", false);

        let pred_hits = super::par_search(data, root, query, k, max_multiplier);
        assert_eq!(
            pred_hits.len(),
            true_hits.len(),
            "Parallel Knn search failed: {pred_hits:?}"
        );
        check_search_by_distance(true_hits, pred_hits, "Par KnnClustered", false);

        true
    }

    #[test]
    fn line() -> Result<(), String> {
        let data = gen_line_data(10)?;
        let query = &0;

        let criteria = |c: &Ball<u32>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::new_tree(&data, &criteria, seed);

        let max_multiplier = 2;

        for k in [1, 4, 8] {
            assert!(check_knn(&ball, &data, query, k, max_multiplier));
        }

        let mut data = data;
        let root = OffsetBall::from_ball_tree(ball, &mut data);

        for k in [1, 4, 8] {
            assert!(check_knn(&root, &data, query, k, max_multiplier));
        }

        Ok(())
    }

    #[test]
    fn grid() -> Result<(), String> {
        let data = gen_grid_data(10)?;
        let query = &(0.0, 0.0);

        let criteria = |c: &Ball<f32>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::new_tree(&data, &criteria, seed);

        let max_multiplier = 2.0;

        for k in [1, 4, 8, 16, 32] {
            assert!(check_knn(&ball, &data, query, k, max_multiplier));
        }

        let mut data = data;
        let root = OffsetBall::from_ball_tree(ball, &mut data);

        for k in [1, 4, 8, 16, 32] {
            assert!(check_knn(&root, &data, query, k, max_multiplier));
        }

        Ok(())
    }
}
