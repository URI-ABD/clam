//! Search a given tree for some given queries.

use std::path::Path;

use abd_clam::cakes::Searchable;
use abd_clam::{Cluster, FlatVec, Metric};
use distances::Number;

use crate::trees::ShellBall;
use crate::{data::ShellFlatVec, metrics::ShellMetric, trees::ShellTree};

pub fn search_tree<P: AsRef<Path>>(
    data_path: P,
    tree_path: P,
    instances_path: P,
    query_algorithms: Vec<crate::search::QueryAlgorithm<f64>>,
    metric: ShellMetric,
) -> Result<(), String> {
    // Load the tree and data
    let tree = ShellTree::read_from(tree_path)?;
    let data = ShellFlatVec::read_from(data_path)?;
    let instances = ShellFlatVec::read_from(instances_path)?;

    match (data, tree, instances, metric) {
        (ShellFlatVec::F32(d), ShellTree::Ball(ShellBall::F32(c)), ShellFlatVec::F32(i), ShellMetric::Euclidean(m)) => {
            search(&d, &m, &c, &i, &query_algorithms)
        }
        (ShellFlatVec::F64(d), ShellTree::Ball(ShellBall::F64(c)), ShellFlatVec::F64(i), ShellMetric::Euclidean(m)) => {
            search(&d, &m, &c, &i, &query_algorithms)
        }
        _ => todo!("Implement support for other data types"),
    };

    Ok(())
}

fn search<I: std::fmt::Debug, T: Number + 'static, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>>(
    data: &D,
    metric: &M,
    root: &C,
    instances: &FlatVec<I, usize>,
    algs: &[crate::search::QueryAlgorithm<f64>],
) {
    for instance in instances.items() {
        println!("{:?}", instance);
        for alg in algs {
            let result = alg.get().search(data, metric, root, &instance);
            println!("Result {}: {:?}", alg, result);
        }
    }
}
