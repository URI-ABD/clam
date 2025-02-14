//! Build the dimension reduction.

use abd_clam::{
    cluster::ParPartition,
    dataset::{AssociatesMetadata, AssociatesMetadataMut},
    mbed::MassSpringSystem,
    metric::ParMetric,
    Dataset, FlatVec,
};

/// Build the dimension reduction.
///
/// # Type Parameters
///
/// - `P`: The type of the path to the output directory.
/// - `I`: The type of the items in the dataset.
/// - `D`: The type of the dataset.
/// - `M`: The type of the metric.
/// - `C`: The type of the root `Cluster`.
/// - `CC`: The type of the criteria function.
#[allow(clippy::too_many_arguments)]
pub fn build<P, I, M, C, CC, Me, const DIM: usize>(
    out_dir: &P,
    data: FlatVec<I, Me>,
    metric: M,
    criteria: &CC,
    dimensions: usize,
    name: &str,
    checkpoint_frequency: usize,
    seed: Option<u64>,
    beta: f32,
    k: f32,
    f: f32,
    min_k: Option<f32>,
    dt: f32,
    patience: usize,
    target: Option<f32>,
    max_steps: Option<usize>,
) -> Result<FlatVec<[f32; DIM], Me>, String>
where
    P: AsRef<std::path::Path>,
    I: Send + Sync,
    M: ParMetric<I, f32>,
    C: ParPartition<f32>,
    CC: (Fn(&C) -> bool) + Send + Sync,
    Me: Clone + Send + Sync,
{
    ftlog::info!("Building the dimension reduction ...");
    ftlog::info!("Output directory: {:?}", out_dir.as_ref());
    ftlog::info!("Dataset: {:?}", data.name());
    ftlog::info!("Metric: {:?}", metric.name());
    ftlog::info!("Dimensions: {dimensions}");
    ftlog::info!("Name: {name}");
    ftlog::info!("Checkpoint frequency: {checkpoint_frequency}");

    ftlog::info!("Creating the tree ...");
    let root = C::par_new_tree_iterative(&data, &metric, criteria, seed, 128);

    ftlog::info!("Starting the dimension reduction ...");
    let system = MassSpringSystem::<DIM, _, _>::from_root(&root, beta, name)
        .par_evolve_to_leaves(&data, &metric, k, f, min_k, dt, patience, target, max_steps, out_dir);

    ftlog::info!("Extracting the reduced embedding ...");
    system.get_reduced_embedding().with_metadata(data.metadata())
}
