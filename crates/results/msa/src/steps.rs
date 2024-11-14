//! Steps for the MSA pipeline.

use std::path::Path;

use abd_clam::{
    adapter::ParBallAdapter, cakes::OffBall, cluster::WriteCsv, msa, partition::ParPartition, Ball, Cluster, Dataset,
    FlatVec, Metric, MetricSpace,
};
use distances::Number;

type Fv<U> = FlatVec<String, U, String>;
type B<U> = Ball<String, U, Fv<U>>;
type Ob<U> = OffBall<String, U, Fv<U>, B<U>>;

/// Build the aligned datasets.
pub fn build_aligned<P: AsRef<Path>>(
    matrix: &crate::CostMatrix,
    off_ball: &Ob<i32>,
    data: &Fv<i32>,
    out_path: P,
) -> Result<(), String> {
    ftlog::info!("Setting up aligner...");
    let cost_matrix = matrix.cost_matrix::<i32>();
    let aligner = if matrix.is_minimizer() {
        msa::Aligner::new_minimizer(&cost_matrix, b'-')
    } else {
        msa::Aligner::new_maximizer(&cost_matrix, b'-')
    };

    ftlog::info!("Aligning sequences...");
    let builder = msa::Columnar::<i32>::new(&aligner).par_with_binary_tree(off_ball, data);

    ftlog::info!("Extracting aligned sequences...");
    let msa = msa::Msa::par_from_columnar(&builder);
    let aligned_sequences = msa.strings();
    let width = builder.width();

    ftlog::info!("Finished aligning {} sequences.", builder.len());
    let metric = Metric::<_, i32>::default();
    let data = FlatVec::new(aligned_sequences, metric)?
        .with_metadata(data.metadata())?
        .with_dim_lower_bound(width)
        .with_dim_upper_bound(width);

    let path = out_path.as_ref();
    ftlog::info!("Writing MSA to {path:?}");
    crate::data::write_fasta(&data, path)?;

    Ok(())
}

/// Read the aligned fasta file.
pub fn read_aligned<P: AsRef<Path>, U: Number>(path: P, metric: Metric<String, U>) -> Result<Fv<U>, String> {
    ftlog::info!("Reading aligned sequences from {:?}", path.as_ref());

    let ([aligned_sequences, _], [width, _]) = results_cakes::data::fasta::read(path, 0, false)?;
    let (metadata, aligned_sequences): (Vec<_>, Vec<_>) = aligned_sequences.into_iter().unzip();
    let data = FlatVec::new(aligned_sequences, metric)?
        .with_metadata(&metadata)?
        .with_dim_lower_bound(width)
        .with_dim_upper_bound(width);

    Ok(data)
}

/// Build the Offset Ball and the dataset.
pub fn build_offset_ball<P: AsRef<Path>>(
    ball: B<i32>,
    data: Fv<i32>,
    ball_path: P,
    data_path: P,
) -> Result<(Ob<i32>, Fv<i32>), String> {
    ftlog::info!("Building Offset Ball and permuted dataset.");
    let (off_ball, data) = OffBall::par_from_ball_tree(ball, data);

    ftlog::info!("Writing MSA ball to {:?}", ball_path.as_ref());
    bincode::serialize_into(std::fs::File::create(ball_path).map_err(|e| e.to_string())?, &off_ball)
        .map_err(|e| e.to_string())?;

    ftlog::info!("Writing MSA data to {:?}", data_path.as_ref());
    bincode::serialize_into(std::fs::File::create(data_path).map_err(|e| e.to_string())?, &data)
        .map_err(|e| e.to_string())?;

    Ok((off_ball, data))
}

/// Read the Offset Ball and the dataset from disk.
pub fn read_offset_ball<P: AsRef<Path>>(
    ball_path: P,
    data_path: P,
    metric: Metric<String, i32>,
) -> Result<(Ob<i32>, Fv<i32>), String> {
    ftlog::info!("Reading MSA ball from {:?}", ball_path.as_ref());
    let off_ball = bincode::deserialize_from(std::fs::File::open(ball_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    ftlog::info!("Reading MSA data from {:?}", data_path.as_ref());
    let data: Fv<i32> = bincode::deserialize_from(std::fs::File::open(data_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    ftlog::info!("Finished reading MSA ball and data.");

    Ok((off_ball, data.with_metric(metric)))
}

/// Build the Ball and the dataset.
pub fn build_ball<P: AsRef<Path>>(data: &Fv<i32>, ball_path: P, csv_path: P) -> Result<B<i32>, String> {
    // Create the ball from scratch.
    ftlog::info!("Building ball.");
    let mut depth = 0;
    let seed = Some(42);

    let indices = (0..data.cardinality()).collect::<Vec<_>>();
    let mut ball = Ball::par_new(data, &indices, 0, seed);
    let depth_delta = ball.max_recursion_depth();

    let criteria = |c: &Ball<_, _, _>| c.depth() < 1;
    ball.par_partition(data, &criteria, seed);

    while ball.leaves().into_iter().any(|c| !c.is_singleton()) {
        depth += depth_delta;
        let criteria = |c: &Ball<_, _, _>| c.depth() < depth;
        ball.par_partition_further(data, &criteria, seed);
    }

    let num_leaves = ball.leaves().len();
    ftlog::info!("Built ball with {num_leaves} leaves.");

    // Serialize the ball to disk.
    let ball_path = ball_path.as_ref();
    ftlog::info!("Writing ball to {ball_path:?}");
    bincode::serialize_into(std::fs::File::create(ball_path).map_err(|e| e.to_string())?, &ball)
        .map_err(|e| e.to_string())?;

    // Write the ball to a CSV file.
    let csv_path = csv_path.as_ref();
    ftlog::info!("Writing ball to CSV at {csv_path:?}");
    ball.write_to_csv(&csv_path)?;

    Ok(ball)
}

/// Read the Ball from disk.
pub fn read_ball<P: AsRef<Path>>(path: P) -> Result<B<i32>, String> {
    ftlog::info!("Reading ball from {:?}", path.as_ref());

    let reader = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let ball = bincode::deserialize_from(reader).map_err(|e| e.to_string())?;

    ftlog::info!("Finished reading Ball.");

    Ok(ball)
}
