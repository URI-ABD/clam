//! Steps for the MSA pipeline.

use std::path::Path;

use abd_clam::{
    cakes::PermutedBall,
    cluster::{adapter::ParBallAdapter, BalancedBall, ClusterIO, Csv, ParPartition},
    dataset::{AssociatesMetadata, AssociatesMetadataMut, DatasetIO},
    metric::ParMetric,
    msa::{self, Aligner, Sequence},
    Ball, Cluster, Dataset, FlatVec,
};

type B<U> = Ball<U>;
type Pb<U> = PermutedBall<U, B<U>>;

/// Build the aligned datasets.
pub fn build_aligned<P: AsRef<Path>>(
    matrix: &crate::CostMatrix,
    gap_open: Option<usize>,
    perm_ball: &Pb<i32>,
    data: &FlatVec<Sequence<i32>, String>,
    out_path: P,
) -> Result<(), String> {
    ftlog::info!("Setting up aligner...");
    let cost_matrix = matrix.cost_matrix::<i32>(gap_open);
    let aligner = msa::Aligner::new(&cost_matrix, b'-');

    ftlog::info!("Aligning sequences...");
    // let builder = msa::Columnar::<i32>::new(&aligner).par_with_binary_tree(perm_ball, data);
    let builder = msa::Columnar::<i32>::new(&aligner).par_with_tree(perm_ball, data);

    ftlog::info!("Extracting aligned sequences...");
    let msa = builder.to_flat_vec_rows().with_metadata(data.metadata())?;
    let transformer = |s: Vec<u8>| s.into_iter().map(|c| c as char).collect::<String>();
    let msa = msa.transform_items(transformer);

    ftlog::info!("Finished aligning {} sequences.", builder.len());
    let data = msa::MSA::new(&aligner, msa)?;

    let path = out_path.as_ref();
    ftlog::info!("Writing MSA to {path:?}");
    crate::data::write_fasta(&data, path)?;

    Ok(())
}

/// Read the aligned fasta file.
pub fn read_aligned<P: AsRef<Path>>(path: &P, aligner: &Aligner<i32>) -> Result<msa::MSA<String, i32, String>, String> {
    ftlog::info!("Reading aligned sequences from {:?}", path.as_ref());

    let ([aligned_sequences, _], [width, _]) = results_cakes::data::fasta::read(path, 0, false)?;
    let (metadata, aligned_sequences): (Vec<_>, Vec<_>) = aligned_sequences.into_iter().unzip();

    let data = FlatVec::new(aligned_sequences)?
        .with_dim_lower_bound(width)
        .with_dim_upper_bound(width)
        .with_metadata(&metadata)?;

    msa::MSA::new(aligner, data)
}

/// Build the `PermutedBall` and the permuted dataset.
#[allow(clippy::type_complexity)]
pub fn build_perm_ball<'a, P: AsRef<Path>, M: ParMetric<Sequence<'a, i32>, i32>>(
    ball: B<i32>,
    data: FlatVec<Sequence<'a, i32>, String>,
    metric: &M,
    ball_path: &P,
    data_path: &P,
) -> Result<(Pb<i32>, FlatVec<Sequence<'a, i32>, String>), String> {
    ftlog::info!("Building PermutedBall and permuted dataset.");
    let (ball, data) = PermutedBall::par_from_ball_tree(ball, data, metric);

    ftlog::info!("Writing PermutedBall to {:?}", ball_path.as_ref());
    ball.write_to(ball_path)?;

    ftlog::info!("Writing PermutedData to {:?}", data_path.as_ref());
    let transformer = |seq: Sequence<'a, i32>| seq.seq().to_string();
    let writable_data = data.clone().transform_items(transformer);
    writable_data.write_to(data_path)?;

    Ok((ball, data))
}

/// Read the `PermutedBall` and the permuted dataset from disk.
#[allow(clippy::type_complexity)]
pub fn read_permuted_ball<'a, P: AsRef<Path>>(
    ball_path: &P,
    data_path: &P,
    aligner: &'a Aligner<i32>,
) -> Result<(Pb<i32>, FlatVec<Sequence<'a, i32>, String>), String> {
    ftlog::info!("Reading PermutedBall from {:?}", ball_path.as_ref());
    let ball = Pb::read_from(ball_path)?;

    ftlog::info!("Reading PermutedData from {:?}", data_path.as_ref());
    let data = FlatVec::<String, String>::read_from(data_path)?;
    let transformer = |s: String| Sequence::new(s, Some(aligner));
    let data = data.transform_items(transformer);

    Ok((ball, data))
}

/// Build the Ball and the dataset.
pub fn build_ball<'a, P: AsRef<Path>, M: ParMetric<Sequence<'a, i32>, i32>>(
    data: &FlatVec<Sequence<'a, i32>, String>,
    metric: &M,
    ball_path: &P,
    csv_path: &P,
) -> Result<B<i32>, String> {
    // Create the ball from scratch.
    ftlog::info!("Building ball.");
    let mut depth = 0;
    let seed = Some(42);

    let indices = (0..data.cardinality()).collect::<Vec<_>>();
    let mut ball = Ball::par_new(data, metric, &indices, 0, seed)
        .unwrap_or_else(|e| unreachable!("We ensured that indices is non-empty: {e}"));
    let depth_delta = abd_clam::utils::max_recursion_depth();

    let criteria = |c: &Ball<_>| c.depth() < 1;
    ball.par_partition(data, metric, &criteria, seed);

    while ball.leaves().into_iter().any(|c| !c.is_singleton()) {
        depth += depth_delta;
        let criteria = |c: &Ball<_>| c.depth() < depth;
        ball.par_partition_further(data, metric, &criteria, seed);
    }

    let num_leaves = ball.leaves().len();
    ftlog::info!("Built ball with {num_leaves} leaves.");

    // Serialize the ball to disk.
    ftlog::info!("Writing ball to {:?}", ball_path.as_ref());
    ball.write_to(ball_path)?;

    // Write the ball to a CSV file.;
    ftlog::info!("Writing ball to CSV at {:?}", csv_path.as_ref());
    ball.write_to_csv(&csv_path)?;

    Ok(ball)
}

/// Read the Ball from disk.
pub fn read_ball<P: AsRef<Path>>(path: &P) -> Result<B<i32>, String> {
    ftlog::info!("Reading ball from {:?}", path.as_ref());
    let ball = Ball::read_from(path)?;
    ftlog::info!("Finished reading Ball.");

    Ok(ball)
}

/// Build the `Ball` with a balanced partition.
pub fn build_balanced_ball<'a, P: AsRef<Path>, M: ParMetric<Sequence<'a, i32>, i32>>(
    data: &FlatVec<Sequence<'a, i32>, String>,
    metric: &M,
    ball_path: &P,
    csv_path: &P,
) -> Result<B<i32>, String> {
    // Create the ball from scratch.
    ftlog::info!("Building Balanced ball.");
    let seed = Some(42);

    let criteria = |c: &BalancedBall<_>| c.cardinality() > 1;
    let ball = BalancedBall::par_new_tree(data, metric, &criteria, seed).into_ball();

    let num_leaves = ball.leaves().len();
    ftlog::info!("Built BalancedBall with {num_leaves} leaves.");

    // Serialize the `BalancedBall` to disk.
    ftlog::info!("Writing BalancedBall to {:?}", ball_path.as_ref());
    ball.write_to(ball_path)?;

    // Write the `BalancedBall` to a CSV file.
    ftlog::info!("Writing BalancedBall to CSV at {:?}", csv_path.as_ref());
    ball.write_to_csv(&csv_path)?;

    Ok(ball)
}
