//! Build the dimension reduction.

use abd_clam::{
    cluster::{BalancedBall, ParPartition},
    dataset::{AssociatesMetadata, AssociatesMetadataMut},
    mbed::MSS,
    metric::ParMetric,
    Ball, Dataset, FlatVec, ParDiskIO,
};
use distances::number::Float;

/// Build the dimension reduction.
///
/// # Type Parameters
///
/// - `P`: The type of the path to the output directory.
/// - `I`: The type of the items in the dataset.
/// - `D`: The type of the dataset.
/// - `M`: The type of the metric.
/// - `Me`: The type of the metadata.
/// - `F`: The type of the floating-point numbers in the reduction.
/// - `DIM`: The number of dimensions.
#[allow(clippy::too_many_arguments)]
pub fn build<P, I, M, Me, F, const DIM: usize>(
    out_dir: &P,
    data: &FlatVec<I, Me>,
    metric: M,
    balanced: bool,
    seed: Option<u64>,
    beta: F,
    k: F,
    dk: F,
    retention_depth: usize,
    f: F,
    dt: F,
    patience: usize,
    target: F,
    max_steps: usize,
) -> Result<FlatVec<[f64; DIM], Me>, String>
where
    P: AsRef<std::path::Path>,
    I: Send + Sync,
    M: ParMetric<I, f32>,
    Me: Clone + Send + Sync,
    F: Float,
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

    // ftlog::info!("Setting up the simulation...");
    // let mut system = MassSpringSystem::<DIM, _, f32, _>::new(data.cardinality())?
    //     .with_metadata(data.metadata())?
    //     .with_beta(beta)?
    //     .with_k(k)?
    //     .with_dk(dk)?
    //     .with_dt(dt)?
    //     .with_f(f)?
    //     .with_retention_depth(retention_depth)
    //     .with_patience(patience)
    //     .with_max_steps(max_steps)
    //     .with_target(target)?;

    // ftlog::info!("Starting the simulation...");
    // system.par_initialize_with_root(&root, data, &metric, &mut rng);
    // let steps = system.par_simulate_to_leaves(data, &metric, &mut rng);

    // // Stack the steps into a single array.
    // let arrays = steps.into_iter().map(|step| step.to_array2()).collect::<Vec<_>>();
    // let arrays = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
    // let stack = ndarray::stack(ndarray::Axis(0), &arrays).map_err(|e| e.to_string())?;
    // let stack_path = out_dir.as_ref().join(format!("{}-stack.npy", data.name()));
    // ndarray_npy::write_npy(&stack_path, &stack).map_err(|e| e.to_string())?;

    // ftlog::info!("Extracting the reduced embedding...");
    // Ok(system.par_extract_positions())

    let drag = F::ONE - beta;
    let ke_threshold = target;
    // let box_len = F::from(10.0);
    let loosening_factor = dk;
    let replace_fraction = f;
    let loosening_threshold = retention_depth;
    let mut system = MSS::<f32, Ball<_>, f64, DIM>::new(
        drag.as_f64(),
        k.as_f64(),
        dt.as_f64(),
        patience,
        ke_threshold.as_f64(),
        max_steps,
        // box_len,
        loosening_factor.as_f64(),
        replace_fraction.as_f64(),
        loosening_threshold,
    )?
    .init_with_root(&mut rng, data, &metric, &root);
    let [ke, pe] = system.simulate_to_leaves(&mut rng, data, &metric);

    // ftlog::info!("Final KE: {:.2e}, PE: {:.2e}", ke.as_f64(), pe.as_f64());
    ftlog::info!("Final KE: {:.2e}, PE: {:.2e}", ke, pe);
    system.extract_positions()?.with_metadata(data.metadata())
}
