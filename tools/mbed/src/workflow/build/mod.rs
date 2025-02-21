//! Build the dimension reduction.

use abd_clam::{
    cluster::{BalancedBall, ParPartition},
    dataset::AssociatesMetadata,
    mbed::MassSpringSystem,
    metric::ParMetric,
    Ball, Dataset, FlatVec, ParDiskIO,
};

/// Build the dimension reduction.
///
/// # Type Parameters
///
/// - `P`: The type of the path to the output directory.
/// - `I`: The type of the items in the dataset.
/// - `D`: The type of the dataset.
/// - `M`: The type of the metric.
/// - `Me`: The type of the metadata.
/// - `DIM`: The number of dimensions.
#[allow(clippy::too_many_arguments)]
pub fn build<P, I, M, Me, const DIM: usize>(
    out_dir: &P,
    data: &FlatVec<I, Me>,
    metric: M,
    balanced: bool,
    seed: Option<u64>,
    beta: f32,
    k: f32,
    dk: f32,
    retention_depth: usize,
    f: f32,
    dt: f32,
    patience: usize,
    target: f32,
    max_steps: usize,
) -> Result<FlatVec<[f32; DIM], Me>, String>
where
    P: AsRef<std::path::Path>,
    I: Send + Sync,
    M: ParMetric<I, f32>,
    Me: Clone + Send + Sync,
{
    ftlog::info!("Building the dimension reduction...");
    ftlog::info!("Output directory: {:?}", out_dir.as_ref());
    ftlog::info!("Dataset: {:?}", data.name());
    ftlog::info!("Metric: {:?}", metric.name());
    ftlog::info!("Dimensions: {DIM}");

    let mut rng = rand::thread_rng();

    ftlog::info!("Creating the tree...");
    let tree_path = out_dir.as_ref().join(format!("{}-tree.bin", data.name()));
    let root = if tree_path.exists() {
        Ball::<f32>::par_read_from(&tree_path)?
    } else {
        let root = if balanced {
            let criteria = |_: &BalancedBall<f32>| true;
            BalancedBall::par_new_tree_iterative(data, &metric, &criteria, seed, 128).into_ball()
        } else {
            let criteria = |_: &Ball<f32>| true;
            Ball::par_new_tree_iterative(data, &metric, &criteria, seed, 128)
        };
        root.par_write_to(&tree_path)?;
        root
    };

    ftlog::info!("Setting up the simulation...");
    let mut system = MassSpringSystem::<DIM, _, f32, _>::new(data)?
        .with_metadata(data.metadata())?
        .with_beta(beta)?
        .with_k(k)?
        .with_dk(dk)?
        .with_dt(dt)?
        .with_f(f)?
        .with_retention_depth(retention_depth)
        .with_patience(patience)
        .with_max_steps(max_steps)
        .with_target(target)?;

    ftlog::info!("Starting the simulation...");
    system.par_initialize_with_root(&root, data, &metric, &mut rng);
    let steps = system.par_simulate_to_leaves(data, &metric, &mut rng);

    // Stack the steps into a single array.
    let arrays = steps.into_iter().map(|step| step.to_array2()).collect::<Vec<_>>();
    let arrays = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
    let stack = ndarray::stack(ndarray::Axis(0), &arrays).map_err(|e| e.to_string())?;
    let stack_path = out_dir.as_ref().join(format!("{}-stack.npy", data.name()));
    ndarray_npy::write_npy(&stack_path, &stack).map_err(|e| e.to_string())?;

    ftlog::info!("Extracting the reduced embedding...");
    Ok(system.par_extract_positions())
}
