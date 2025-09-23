//! Build the dimension reduction.

use abd_clam::{Ball, DistanceValue, FloatDistanceValue, ParClamIO, ParPartition, mbed::Complex};
use ndarray::prelude::*;
use num::traits::{FromBytes, ToBytes};
use rand::distr::uniform::SampleUniform;

use crate::{
    data::ShellData,
    metrics::{Metric, cosine, euclidean, levenshtein},
    npy,
};

/// Build the dimension reduction.
///
/// # Type Parameters
///
/// - `P`: The type of the path to the output directory.
/// - `F`: The type of the floating-point numbers in the reduction.
/// - `DIM`: The number of dimensions.
#[allow(clippy::too_many_arguments)]
pub fn build_new_embedding<P, F, const DIM: usize>(
    out_dir: &P,
    data: &ShellData,
    metric: &Metric,
    beta: F,
    k: F,
    dk: F,
    dt: F,
    patience: usize,
    target: F,
    max_steps: usize,
) -> Result<Vec<[f32; DIM]>, String>
where
    P: AsRef<std::path::Path>,
    F: FloatDistanceValue + Send + Sync + std::fmt::Debug + SampleUniform,
{
    ftlog::info!("Building the dimension reduction...");
    ftlog::info!("Output directory: {:?}", out_dir.as_ref());
    ftlog::info!("Dimensions: {DIM}");

    match metric {
        Metric::Levenshtein => match data {
            ShellData::String(data) => {
                let (data, _) = data.iter().cloned().unzip::<_, _, Vec<_>, Vec<_>>();
                build_generic::<_, _, u32, _, _, DIM, 4>(
                    out_dir,
                    &data,
                    &levenshtein,
                    beta,
                    k,
                    dk,
                    dt,
                    patience,
                    target,
                    max_steps,
                )
            }
            _ => Err("The Levenshtein metric can only be used with string data.".to_string()),
        },
        Metric::Euclidean => match data {
            ShellData::String(_) => Err("The Euclidean metric cannot be used with string data.".to_string()),
            ShellData::F32(data) => build_generic::<_, _, f32, _, _, DIM, 4>(
                out_dir, data, &euclidean, beta, k, dk, dt, patience, target, max_steps,
            ),
            ShellData::F64(data) => build_generic::<_, _, f64, _, _, DIM, 8>(
                out_dir, data, &euclidean, beta, k, dk, dt, patience, target, max_steps,
            ),
            _ => {
                todo!("Implement remaining match arms")
            }
        },
        Metric::Cosine => match data {
            ShellData::String(_) => Err("The Cosine metric cannot be used with string data.".to_string()),
            ShellData::F32(data) => build_generic::<_, _, f32, _, _, DIM, 4>(
                out_dir, data, &cosine, beta, k, dk, dt, patience, target, max_steps,
            ),
            ShellData::F64(data) => build_generic::<_, _, f64, _, _, DIM, 8>(
                out_dir, data, &cosine, beta, k, dk, dt, patience, target, max_steps,
            ),
            _ => {
                todo!("Implement remaining match arms")
            }
        },
    }
}

/// Generic helper for building the dimension reduction.
///
/// # Type Parameters
///
/// - `P`: The type of the path to the output directory.
/// - `I`: The type of the items in the dataset.
/// - `T`: The type of the distance values.
/// - `M`: The type of the distance metric.
/// - `Me`: The type of the metadata with the dataset.
/// - `F`: The type of the floating-point numbers in the reduction.
/// - `DIM`: The number of dimensions.
/// - `N`: The number of bytes in the distance value type.
#[allow(clippy::too_many_arguments)]
fn build_generic<P, I, T, M, F, const DIM: usize, const N: usize>(
    out_dir: &P,
    data: &Vec<I>,
    metric: &M,
    beta: F,
    k: F,
    dk: F,
    dt: F,
    patience: usize,
    target: F,
    max_steps: usize,
) -> Result<Vec<[f32; DIM]>, String>
where
    P: AsRef<std::path::Path>,
    I: Send + Sync,
    T: DistanceValue + ToBytes<Bytes = [u8; N]> + FromBytes<Bytes = [u8; N]> + Send + Sync,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    F: FloatDistanceValue + Send + Sync + std::fmt::Debug + SampleUniform,
{
    let mut rng = rand::rng();

    ftlog::info!("Creating the tree...");
    let tree_path = out_dir.as_ref().join("tree.bin");
    let root = if tree_path.exists() {
        Ball::<T>::par_read_from(&tree_path)?
    } else {
        let root = Ball::par_new_tree_iterative(data, metric, &|_| true, 128);
        root.par_write_to(&tree_path)?;
        root
    };

    ftlog::info!("Setting up the simulation...");
    let drag_coefficient = F::one() - beta;
    let spring_constant = k;
    let loosening_factor = dk;
    let mut system = Complex::<_, _, F, DIM>::new(&root, drag_coefficient, spring_constant, loosening_factor);

    ftlog::info!("Running the simulation...");
    let tolerance = target;
    let n = patience;
    let checkpoints = system.par_simulate_to_leaves(&mut rng, data, metric, max_steps, tolerance, dt, n);

    ftlog::info!("Writing the energy history...");
    let energy_history = system
        .energy_history()
        .iter()
        .map(|&(ke, pe)| {
            [
                ke.to_f32()
                    .unwrap_or_else(|| unreachable!("Could not convert kinetic energy to f32")),
                pe.to_f32()
                    .unwrap_or_else(|| unreachable!("Could not convert potential energy to f32")),
            ]
        })
        .collect::<Vec<_>>();
    let energy_history = npy::to_array2(&energy_history)?;
    let energy_path = out_dir.as_ref().join("energy.npy");
    ndarray_npy::write_npy(&energy_path, &energy_history).map_err(|e| e.to_string())?;

    ftlog::info!("Writing the {} checkpoints...", checkpoints.len());
    let steps = checkpoints
        .into_iter()
        .map(|step| {
            step.into_iter()
                .map(|row| {
                    let mut ret = [0.0; DIM];
                    for (a, b) in ret.iter_mut().zip(row.iter()) {
                        *a = b
                            .to_f32()
                            .unwrap_or_else(|| unreachable!("Could not convert coordinate to f32"));
                    }
                    ret
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let final_step = steps.last().cloned().ok_or("Simulation produced no steps.".to_string());

    let arrays = steps
        .into_iter()
        .map(|step| npy::to_array2(&step))
        .collect::<Result<Vec<_>, _>>()?;
    let arrays = arrays.iter().map(ArrayBase::view).collect::<Vec<_>>();
    let stack = ndarray::stack(ndarray::Axis(0), &arrays).map_err(|e| e.to_string())?;
    let stack_path = out_dir.as_ref().join("stack.npy");
    ndarray_npy::write_npy(&stack_path, &stack).map_err(|e| e.to_string())?;

    ftlog::info!("Returning the resulting embedding...");
    final_step
}
