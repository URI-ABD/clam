//! Data of aligned sequences.

use std::collections::HashMap;

use abd_clam::{
    adapter::{ParAdapter, ParBallAdapter},
    cakes::{Algorithm, CodecData, Decompressible, OffBall, SquishyBall},
    partition::ParPartition,
    Ball, Cluster, Dataset, FlatVec, MetricSpace, WriteCsv,
};
use distances::Number;

use super::{instances::Aligned, PathManager};

type I = Aligned<u32>;
type U = u32;
type M = String;
type Co = FlatVec<I, U, M>;
type B = Ball<I, U, Co>;
type Dec = CodecData<I, U, M>;
type Sb = SquishyBall<I, U, Co, Dec, B>;
type Hits = Vec<Vec<(usize, U)>>;

/// The group of types used for the datasets of aligned sequences.
pub struct Group {
    path_manager: PathManager,
    uncompressed: Co,
    ball: B,
    compressed: Dec,
    squishy_ball: Sb,
    #[allow(dead_code)]
    query_ids: Vec<M>,
    queries: Vec<I>,
}

impl Group {
    /// Creates a new group of datasets and trees for benchmarks with aligned sequences.
    ///
    /// # Errors
    ///
    /// - If there is an error deserializing or serializing the data.
    /// - If there is an error reading/writing serialized data to/from disk.
    /// - If there is an error writing the trees to csv files.
    pub fn new(path_manager: PathManager, uncompressed: Co, queries: Vec<(M, I)>) -> Result<Self, String> {
        let query_path = path_manager.queries_path();
        if !query_path.exists() {
            // Serialize the queries to disk.
            bincode::serialize_into(std::fs::File::create(&query_path).map_err(|e| e.to_string())?, &queries)
                .map_err(|e| e.to_string())?;
        }
        let (query_ids, queries) = queries.into_iter().unzip();

        let ball_path = path_manager.ball_path();
        let ball = if ball_path.exists() {
            // Deserialize the ball from disk.
            ftlog::info!("Reading ball from {ball_path:?}");
            bincode::deserialize_from(std::fs::File::open(&ball_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?
        } else {
            // Create the ball from scratch.
            ftlog::info!("Creating ball.");
            let mut depth = 0;
            let seed = Some(42);

            let indices = (0..uncompressed.cardinality()).collect::<Vec<_>>();
            let mut ball = Ball::par_new(&uncompressed, &indices, 0, seed);
            let depth_delta = ball.max_recursion_depth();

            let criteria = |c: &Ball<_, _, _>| c.depth() < 1;
            ball.par_partition(&uncompressed, &criteria, seed);

            while ball.leaves().into_iter().any(|c| !c.is_singleton()) {
                depth += depth_delta;
                let criteria = |c: &Ball<_, _, _>| c.depth() < depth;
                ball.par_partition_further(&uncompressed, &criteria, seed);
            }

            // Serialize the ball to disk.
            ftlog::info!("Writing ball to {ball_path:?}");
            bincode::serialize_into(std::fs::File::create(&ball_path).map_err(|e| e.to_string())?, &ball)
                .map_err(|e| e.to_string())?;

            let num_leaves = ball.leaves().len();
            ftlog::info!("Ball has {num_leaves} leaves.");

            // Write the ball to a CSV file.
            ftlog::info!("Writing ball to CSV.");
            ball.write_to_csv(&path_manager.ball_csv_path())?;

            ball
        };

        let squishy_ball_path = path_manager.squishy_ball_path();
        let compressed_path = path_manager.compressed_path();

        let (squishy_ball, mut compressed) = if squishy_ball_path.exists() && compressed_path.exists() {
            ftlog::info!("Reading squishy ball from {squishy_ball_path:?}");
            let squishy_ball =
                bincode::deserialize_from(std::fs::File::open(&squishy_ball_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;

            ftlog::info!("Reading compressed dataset from {compressed_path:?}");
            let codec_data =
                bincode::deserialize_from(std::fs::File::open(&compressed_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;

            (squishy_ball, codec_data)
        } else {
            ftlog::info!("Creating squishy ball and permuted dataset.");
            let (mut squishy_ball, perm_data) = {
                let (off_ball, data) = OffBall::par_from_ball_tree(ball.clone(), uncompressed.clone());
                let mut squishy_ball = SquishyBall::par_adapt_tree_iterative(off_ball, None);

                // Set the costs of the squishy ball and write it to a CSV file.
                ftlog::info!("Setting costs and writing pre-trim ball to CSV.");
                squishy_ball.par_set_costs(&data);
                squishy_ball.write_to_csv(&path_manager.pre_trim_csv_path())?;

                (squishy_ball, data)
            };

            // Trim the squishy ball and write it to a CSV file.
            ftlog::info!("Trimming squishy ball and writing to CSV.");
            squishy_ball.trim(4);
            squishy_ball.write_to_csv(&path_manager.squishy_csv_path())?;

            let num_leaves = squishy_ball.leaves().len();
            ftlog::info!("Built squishy ball with {num_leaves} leaves.");

            assert!(num_leaves > 1, "Squishy ball has only one leaf.");

            // Create the compressed dataset and set its metadata.
            ftlog::info!("Creating compressed dataset.");
            let codec_data = CodecData::from_compressible(&perm_data, &squishy_ball)
                .with_metadata(uncompressed.metadata().to_vec())?;

            let num_bytes = codec_data
                .leaf_bytes()
                .iter()
                .map(|(_, bytes)| core::mem::size_of::<usize>() + bytes.len())
                .sum::<usize>();
            ftlog::info!("Built compressed dataset with {num_bytes} leaf bytes.");

            // Change the metadata type of the squishy ball to match the compressed dataset.
            let squishy_ball = squishy_ball.with_metadata_type::<String>();

            // Serialize the squishy ball and the compressed dataset to disk.
            ftlog::info!("Writing squishy ball to {squishy_ball_path:?}");
            bincode::serialize_into(
                std::fs::File::create(&squishy_ball_path).map_err(|e| e.to_string())?,
                &squishy_ball,
            )
            .map_err(|e| e.to_string())?;

            ftlog::info!("Writing compressed dataset to {compressed_path:?}");
            bincode::serialize_into(
                std::fs::File::create(&compressed_path).map_err(|e| e.to_string())?,
                &codec_data,
            )
            .map_err(|e| e.to_string())?;

            (squishy_ball, codec_data)
        };
        compressed.set_metric(I::metric());

        Ok(Self {
            path_manager,
            uncompressed,
            ball,
            compressed,
            squishy_ball,
            query_ids,
            queries,
        })
    }

    /// Run benchmarks for compressive search on the dataset.
    ///
    /// # Errors
    ///
    /// - If there is an error writing the times to disk.
    pub fn bench_compressive_search(&mut self, num_queries: usize) -> Result<(), String> {
        self.uncompressed.set_metric(Aligned::levenshtein_metric());
        self.compressed.set_metric(Aligned::levenshtein_metric());

        let radius = 5;
        let k = 10;
        let algorithms = [
            // Algorithm::RnnLinear(radius),
            Algorithm::RnnClustered(radius),
            // Algorithm::KnnLinear(k),
            Algorithm::KnnRepeatedRnn(k, 2),
            Algorithm::KnnBreadthFirst(k),
            Algorithm::KnnDepthFirst(k),
        ];

        let num_queries = num_queries.min(self.queries.len());
        let queries = &self.queries[..num_queries];
        ftlog::info!(
            "Running benchmarks for compressive search on {num_queries} queries with {} algorithms",
            algorithms.len()
        );

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
                "Algorithm {} took {:.3e} seconds per query compressed time on {}",
                alg.name(),
                compressed_time,
                self.path_manager.name()
            );

            self.verify_hits(uncompressed_hits, compressed_hits)?;

            let slowdown = compressed_time / uncompressed_time;
            times.insert(
                alg.name(),
                (
                    format!("uncompressed: {uncompressed_time:.4e}"),
                    format!("uncompressed_throughput: {:.4e}", 1.0 / uncompressed_time),
                    format!("compressed: {compressed_time:.4e}"),
                    format!("compressed_throughput: {:.4e}", 1.0 / compressed_time),
                    format!("slowdown: {slowdown:.4}"),
                ),
            );
        }

        ftlog::info!("Writing times to disk.");
        serde_json::to_writer_pretty(
            std::fs::File::create(self.path_manager.times_path()).map_err(|e| e.to_string())?,
            &times,
        )
        .map_err(|e| e.to_string())?;

        self.compressed.set_metric(Aligned::metric());
        self.uncompressed.set_metric(Aligned::metric());

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
