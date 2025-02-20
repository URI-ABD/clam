//! Build the dimension reduction.

use abd_clam::{
    cluster::ParPartition, dataset::AssociatesMetadata, mbed::MassSpringSystem, metric::ParMetric, Dataset, FlatVec,
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
/// - `Me`: The type of the metadata.
/// - `DIM`: The number of dimensions.
#[allow(clippy::too_many_arguments)]
pub fn build<P, I, M, C, CC, Me, const DIM: usize>(
    out_dir: &P,
    data: FlatVec<I, Me>,
    metric: M,
    criteria: &CC,
    name: &str,
    seed: Option<u64>,
    beta: Option<f32>,
    k: f32,
    dk: Option<f32>,
    retention_depth: Option<usize>,
    f: Option<f32>,
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
    ftlog::info!("Building the dimension reduction...");
    ftlog::info!("Output directory: {:?}", out_dir.as_ref());
    ftlog::info!("Dataset: {:?}", data.name());
    ftlog::info!("Metric: {:?}", metric.name());
    ftlog::info!("Dimensions: {DIM}");
    ftlog::info!("Name: {name}");

    let mut rng = rand::thread_rng();

    ftlog::info!("Creating the tree...");
    let root = C::par_new_tree_iterative(&data, &metric, criteria, seed, 128);

    ftlog::info!("Setting up the simulation...");
    let mut system = MassSpringSystem::<DIM, _, f32, C>::new(&data)?
        .with_metadata(data.metadata())?
        .with_beta(beta.unwrap_or(0.99))?
        .with_k(k)?
        .with_dk(dk.unwrap_or(0.5))?
        .with_dt(dt)?
        .with_f(f.unwrap_or(0.5))?
        .with_retention_depth(retention_depth.unwrap_or(4))
        .with_patience(patience)
        .with_max_steps(max_steps.unwrap_or(10_000))
        .with_target(target.unwrap_or(1e-3))?;

    ftlog::info!("Starting the simulation...");
    system.par_initialize_with_root(&root, &data, &metric, &mut rng);
    let steps = system.par_simulate_to_leaves(&data, &metric, &mut rng);

    // Stack the steps into a single array.
    let arrays = steps.into_iter().map(|step| step.to_array2()).collect::<Vec<_>>();
    let arrays = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
    let stack = ndarray::stack(ndarray::Axis(0), &arrays).map_err(|e| e.to_string())?;
    let stack_path = out_dir.as_ref().join(format!("{name}-stack.npy"));
    ndarray_npy::write_npy(&stack_path, &stack).map_err(|e| e.to_string())?;

    ftlog::info!("Extracting the reduced embedding...");
    Ok(system.par_extract_positions())
}
