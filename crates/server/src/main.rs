#[macro_use]
extern crate rocket;
use abd_clam::{
    adapter::ParBallAdapter,
    cakes::{cluster::ParSearchable, cluster::Searchable, Algorithm, CodecData, SquishyBall},
    partition::ParPartition,
    Ball, Cluster, FlatVec, Metric, MetricSpace,
};
use bincode;
use distances;
use rocket::serde::json::Json;
use symagen;

type CodecDataType = CodecData<String, u16, usize>;
type SquishyBallType = SquishyBall<
    String,
    u16,
    FlatVec<String, u16, String>,
    CodecDataType,
    Ball<String, u16, FlatVec<String, u16, String>>,
>;

struct Config {
    squishy_ball_path: String,
    codec_data_path: String,
    distance_fn: fn(&String, &String) -> u16,
}

impl Config {
    /// Create a new configuration from environment variables.
    fn new() -> Self {
        let squishy_ball_path = std::env::var("CLAM_SQUISHY_BALL_PATH").expect("CLAM_SQUISHY_BALL_PATH must be set");
        let codec_data_path = std::env::var("CLAM_CODEC_DATA_PATH").expect("CLAM_CODEC_DATA_PATH must be set");
        Self {
            squishy_ball_path,
            codec_data_path,
            distance_fn: |a: &String, b: &String| distances::strings::levenshtein::<u16>(a, b),
        }
    }
    /// Load the SquishyBall and CodecData from disk.
    fn load(&self) -> (SquishyBallType, CodecDataType) {
        let squishy_ball: SquishyBallType =
            bincode::deserialize_from(std::fs::File::open(&self.squishy_ball_path).unwrap()).unwrap();
        let mut codec_data: CodecDataType =
            bincode::deserialize_from(std::fs::File::open(&self.codec_data_path).unwrap()).unwrap();
        codec_data.set_metric(Metric::new(self.distance_fn, true));
        (squishy_ball, codec_data)
    }
}

#[get("/")]
fn index() -> &'static str {
    "Welcome to the CLAM server."
}

#[get("/search/rnn-clustered?<query>&<radius>")]
fn search_rnn_clustered(
    query: String,
    radius: u16,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let alg = Algorithm::RnnClustered(radius);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, &query, alg);
    Json(results)
}

#[get("/search/knn-repeated-rnn?<query>&<k>&<max_radius_multiplier>")]
fn search_knn_repeated_rnn(
    query: String,
    k: usize,
    max_radius_multiplier: Option<u16>,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let r_m = max_radius_multiplier.unwrap_or(2);
    let alg = Algorithm::KnnRepeatedRnn(k, r_m);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, &query, alg);
    Json(results)
}

#[get("/search/knn-breadth-first?<query>&<k>")]
fn search_knn_breadth_first(
    query: String,
    k: usize,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let alg = Algorithm::KnnBreadthFirst(k);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, &query, alg);
    Json(results)
}

#[get("/search/knn-depth-first?<query>&<k>")]
fn search_knn_depth_first(
    query: String,
    k: usize,
    squishy_ball: &rocket::State<SquishyBallType>,
    codec_data: &rocket::State<CodecDataType>,
) -> Json<Vec<(usize, u16)>> {
    let alg = Algorithm::KnnDepthFirst(k);
    let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, &query, alg);
    Json(results)
}

#[launch]
fn rocket() -> _ {
    let config = match std::env::var("CLAM_BOOTSTRAP") {
        Ok(_) => bootstrap(),
        Err(_) => Config::new(),
    };
    let (squishy_ball, codec_data) = config.load();
    rocket::build().manage(squishy_ball).manage(codec_data).mount(
        "/",
        routes![
            index,
            search_rnn_clustered,
            search_knn_repeated_rnn,
            search_knn_breadth_first,
            search_knn_depth_first,
        ],
    )
}

/// Bootstrap the server by generating a dataset, building a tree, and serializing the tree and dataset to disk.
///
/// Tests the generated data by performing a search on the compressed data before/after serialization.
/// Logs the results of the search so you can use the query in additional testing.
fn bootstrap() -> Config {
    let alphabet = "ACTGN".chars().collect::<Vec<_>>();
    let seed_length = 100;
    let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
    let penalties = distances::strings::Penalties::default();
    let num_clumps = 10;
    let clump_size = 10;
    let clump_radius = 3_u32;
    let inter_clump_distance_range = (clump_radius * 5, clump_radius * 7);
    let len_delta = seed_length / 10;
    let (metadata, data) = symagen::random_edits::generate_clumped_data(
        &seed_string,
        penalties,
        &alphabet,
        num_clumps,
        clump_size,
        clump_radius,
        inter_clump_distance_range,
        len_delta,
    )
    .into_iter()
    .unzip::<_, _, Vec<_>, Vec<_>>();

    // The dataset will use the `levenshtein` distance function from the `distances` crate.
    let distance_fn = |a: &String, b: &String| distances::strings::levenshtein::<u16>(a, b);
    let metric = Metric::new(distance_fn, true);
    let data = FlatVec::new(data, metric.clone())
        .unwrap()
        .with_metadata(metadata.clone())
        .unwrap();

    // We can serialize the dataset to disk without compression.
    let temp_dir = std::env::current_dir().unwrap().join("data");
    if !temp_dir.exists() {
        std::fs::create_dir(&temp_dir).unwrap();
    }
    let flat_path = temp_dir.as_path().join("strings.flat_vec");
    let mut file = std::fs::File::create(&flat_path).unwrap();
    bincode::serialize_into(&mut file, &data).unwrap();

    // We build a tree from the dataset.
    let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
    let seed = Some(42);
    let ball = Ball::par_new_tree(&data, &criteria, seed);

    // We can serialize the tree to disk.
    let ball_path = temp_dir.as_path().join("strings.ball");
    let mut file = std::fs::File::create(&ball_path).unwrap();
    bincode::serialize_into(&mut file, &ball).unwrap();

    // We can adapt the tree and dataset to allow for compression and compressed search.
    let (squishy_ball, codec_data) = SquishyBall::par_from_ball_tree(ball, data);

    // We can serialize the compressed dataset to disk.
    let codec_path = temp_dir.as_path().join("strings.codec_data");
    let mut file = std::fs::File::create(&codec_path).unwrap();
    bincode::serialize_into(&mut file, &codec_data).unwrap();

    // We can serialize the compressed tree to disk.
    let squishy_ball_path = temp_dir.as_path().join("strings.squishy_ball");
    let mut file = std::fs::File::create(&squishy_ball_path).unwrap();
    bincode::serialize_into(&mut file, &squishy_ball).unwrap();

    let alg = Algorithm::RnnClustered(2);
    let results = squishy_ball.par_search(&codec_data, &seed_string, alg);
    println!("{:?}, {:?} = {:?}", squishy_ball, seed_string, results);

    let config = Config {
        squishy_ball_path: squishy_ball_path.to_str().unwrap().to_string(),
        codec_data_path: codec_path.to_str().unwrap().to_string(),
        distance_fn,
    };
    let (squishy_ball, codec_data) = config.load();
    let results = squishy_ball.search(&codec_data, &seed_string, alg);
    println!("{:?}, {:?} = {:?}", squishy_ball, seed_string, results);

    config
}
