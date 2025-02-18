//! Build the dimension reduction.

use abd_clam::{cluster::ParPartition, mbed::MassSpringSystem, metric::ParMetric, Dataset, FlatVec};

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
    dimensions: usize,
    name: &str,
    checkpoint_frequency: usize,
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
    ftlog::info!("Building the dimension reduction ...");
    ftlog::info!("Output directory: {:?}", out_dir.as_ref());
    ftlog::info!("Dataset: {:?}", data.name());
    ftlog::info!("Metric: {:?}", metric.name());
    ftlog::info!("Dimensions: {dimensions}");
    ftlog::info!("Name: {name}");
    ftlog::info!("Checkpoint frequency: {checkpoint_frequency}");

    let mut rng = rand::thread_rng();

    ftlog::info!("Creating the tree ...");
    let root = C::par_new_tree_iterative(&data, &metric, criteria, seed, 128);

    ftlog::info!("Starting the dimension reduction ...");
    let mut system = MassSpringSystem::<DIM, _, f32, C>::new(&data, beta)?;
    system.par_initialize_with_root(k, &root, &data, &metric, &mut rng)?;
    let steps = system.par_simulate_to_leaves(
        k,
        &data,
        &metric,
        &mut rng,
        dt,
        patience,
        target,
        max_steps,
        dk,
        retention_depth,
        f,
    )?;

    // Stack the steps into a single array.
    let arrays = steps.into_iter().map(|step| step.to_array2()).collect::<Vec<_>>();
    let arrays = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
    let stack = ndarray::stack(ndarray::Axis(0), &arrays).map_err(|e| e.to_string())?;
    let stack_path = out_dir.as_ref().join(format!("{name}-stack.npy"));
    ndarray_npy::write_npy(&stack_path, &stack).map_err(|e| e.to_string())?;

    ftlog::info!("Extracting the reduced embedding ...");
    Ok(system.par_extract_positions())
}
