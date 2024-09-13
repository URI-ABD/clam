//! Data of named member sets from the ANN-benchmarks repository.

use std::collections::HashMap;

use abd_clam::{
    adapter::{ParAdapter, ParBallAdapter},
    cakes::{Algorithm, CodecData, OffBall, SquishyBall},
    partition::ParPartition,
    BalancedBall, Ball, Cluster, Dataset, FlatVec, WriteCsv,
};
use distances::Number;

use super::{instances::MemberSet, PathManager};

type I = MemberSet<usize, f32>;
type U = f32;
type M = usize;
type Co = FlatVec<I, U, M>;
type B = Ball<I, U, Co>;
type Dec = CodecData<I, U, M>;
type Sb = SquishyBall<I, U, Co, Dec, B>;
type Hits = Vec<Vec<(usize, U)>>;

/// The group of types used for the datasets of named member sets.
pub struct Group {
    path_manager: PathManager,
    uncompressed: Co,
    ball: B,
    compressed: Dec,
    squishy_ball: Sb,
    #[allow(dead_code)]
    query_ids: Vec<M>,
    queries: Vec<I>,
    #[allow(dead_code)]
    ground_truth: Hits,
}

impl Group {
    /// Creates a new group of datasets and trees for benchmarks with named member sets.
    ///
    /// # Errors
    ///
    /// - If there is an error deserializing or serializing the data.
    /// - If there is an error reading/writing serialized data to/from disk.
    /// - If there is an error writing the trees to csv files.
    pub fn new(
        path_manager: PathManager,
        uncompressed: Co,
        queries: Vec<(M, I)>,
        ground_truth: Hits,
    ) -> Result<Self, String> {
        let query_path = path_manager.queries_path();
        if !query_path.exists() {
            // Serialize the queries to disk.
            bincode::serialize_into(std::fs::File::create(&query_path).map_err(|e| e.to_string())?, &queries)
                .map_err(|e| e.to_string())?;
        }
        let (query_ids, queries) = queries.into_iter().unzip();

        let gt_path = path_manager.ground_truth_path();
        if !gt_path.exists() {
            // Serialize the ground truth to disk.
            bincode::serialize_into(
                std::fs::File::create(&gt_path).map_err(|e| e.to_string())?,
                &ground_truth,
            )
            .map_err(|e| e.to_string())?;
        }

        let ball_path = path_manager.ball_path();
        let ball = if ball_path.exists() {
            // Deserialize the ball from disk.
            bincode::deserialize_from(std::fs::File::open(&ball_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?
        } else {
            let mut max_depth = 0;
            let seed = Some(42);

            let indices = (0..uncompressed.cardinality()).collect::<Vec<_>>();
            let mut ball = BalancedBall::par_new(&uncompressed, &indices, 0, seed);
            let depth_delta = ball.max_recursion_depth();

            loop {
                max_depth += depth_delta;
                let criteria =
                    |c: &BalancedBall<_, _, _>| (c.depth() < max_depth && c.radius() > 0.75) || c.cardinality() > 1_000;
                ball.par_partition_further(&uncompressed, &criteria, seed);

                // If there are no leaves at the current maximum depth, break
                if !ball.leaves().into_iter().any(|c| c.depth() == max_depth) {
                    break;
                }
            }

            let mut ball = Ball::par_from_balanced_ball(ball);

            for leaf in ball.leaves_mut() {
                max_depth = leaf.depth();
                loop {
                    max_depth += depth_delta;
                    let criteria = |c: &Ball<_, _, _>| c.depth() < max_depth;
                    leaf.par_partition_further(&uncompressed, &criteria, seed);

                    // If there are no leaves at the current maximum depth, break
                    if !leaf.leaves().into_iter().any(|c| c.depth() == max_depth) {
                        break;
                    }
                }
            }

            // Serialize the ball to disk.
            bincode::serialize_into(std::fs::File::create(&ball_path).map_err(|e| e.to_string())?, &ball)
                .map_err(|e| e.to_string())?;

            // Write the ball to a CSV file.
            ball.write_to_csv(&path_manager.ball_csv_path())?;

            ball
        };

        let squishy_ball_path = path_manager.squishy_ball_path();
        let compressed_path = path_manager.compressed_path();

        let (squishy_ball, compressed) = if squishy_ball_path.exists() && compressed_path.exists() {
            let squishy_ball =
                bincode::deserialize_from(std::fs::File::open(&squishy_ball_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;

            let codec_data =
                bincode::deserialize_from(std::fs::File::open(&compressed_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;

            (squishy_ball, codec_data)
        } else {
            let (mut squishy_ball, perm_data) = {
                let (off_ball, data) = OffBall::par_from_ball_tree(ball.clone(), uncompressed.clone());
                let mut squishy_ball = SquishyBall::par_adapt_tree_iterative(off_ball, None);

                // Set the costs of the squishy ball and write it to a CSV file.
                squishy_ball.par_set_costs(&data);
                squishy_ball.write_to_csv(&path_manager.pre_trim_csv_path())?;

                (squishy_ball, data)
            };

            // Trim the squishy ball and write it to a CSV file.
            squishy_ball.trim();
            squishy_ball.write_to_csv(&path_manager.squishy_csv_path())?;

            // Create the compressed dataset and set its metadata.
            let codec_data = CodecData::from_compressible(&perm_data, &squishy_ball)
                .with_metadata(uncompressed.metadata().to_vec())?;

            // Serialize the squishy ball and the compressed dataset to disk.
            bincode::serialize_into(
                std::fs::File::create(&squishy_ball_path).map_err(|e| e.to_string())?,
                &squishy_ball,
            )
            .map_err(|e| e.to_string())?;
            bincode::serialize_into(
                std::fs::File::create(&compressed_path).map_err(|e| e.to_string())?,
                &codec_data,
            )
            .map_err(|e| e.to_string())?;

            (squishy_ball, codec_data)
        };

        Ok(Self {
            path_manager,
            uncompressed,
            ball,
            compressed,
            squishy_ball,
            query_ids,
            queries,
            ground_truth,
        })
    }

    /// Run benchmarks for compressive search on the dataset.
    ///
    /// # Errors
    ///
    /// - If there is an error writing the times to disk.
    pub fn bench_compressive_search(&self, num_queries: usize) -> Result<(), String> {
        let radius = 5_f32;
        let k = 10;
        let algorithms = [
            Algorithm::RnnLinear(radius),
            Algorithm::RnnClustered(radius),
            Algorithm::KnnLinear(k),
            Algorithm::KnnRepeatedRnn(k, 2_f32),
            Algorithm::KnnBreadthFirst(k),
            Algorithm::KnnDepthFirst(k),
        ];

        let queries = &self.queries[..num_queries];

        let mut times = HashMap::new();
        for (i, alg) in algorithms.iter().enumerate() {
            ftlog::info!(
                "Running algorithm {} ({}/{}) on {}",
                alg.name(),
                i + 1,
                algorithms.len(),
                self.path_manager.name()
            );

            let uncompressed_start = std::time::Instant::now();
            let uncompressed_hits = alg.par_batch_par_search(&self.uncompressed, &self.ball, queries);
            let uncompressed_time = uncompressed_start.elapsed().as_secs_f32() / num_queries.as_f32();
            ftlog::info!(
                "Algorithm {} took {:.3e} seconds per query uncompressed time on {}",
                alg.name(),
                uncompressed_time,
                self.path_manager.name()
            );

            let compressed_start = std::time::Instant::now();
            let compressed_hits = alg.par_batch_par_search(&self.compressed, &self.squishy_ball, queries);
            let compressed_time = compressed_start.elapsed().as_secs_f32() / num_queries.as_f32();
            ftlog::info!(
                "Algorithm {} took {:.3e} seconds per query uncompressed time on {}",
                alg.name(),
                uncompressed_time,
                self.path_manager.name()
            );

            self.verify_hits(uncompressed_hits, compressed_hits)?;

            times.insert(alg.name(), (uncompressed_time, compressed_time));
        }

        serde_json::to_writer_pretty(
            std::fs::File::create(self.path_manager.times_path()).map_err(|e| e.to_string())?,
            &times,
        )
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Checks that the hits from the uncompressed and compressed datasets are the same.
    #[allow(
        dead_code,
        unused_variables,
        clippy::unnecessary_wraps,
        clippy::needless_pass_by_value,
        clippy::unused_self
    )]
    fn verify_hits(&self, uncompressed: Hits, compressed: Hits) -> Result<(), String> {
        ftlog::warn!("Hit verification not yet implemented.");
        Ok(())
    }
}
