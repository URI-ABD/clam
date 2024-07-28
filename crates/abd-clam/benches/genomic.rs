// use abd_clam::{
//     cakes::Algorithm, cakes::ParSearchable, cakes::Searchable, Ball, Cluster, FlatVec, Metric, ParPartition,
// };
use criterion::*;
// use rand::prelude::*;

// const METRICS: &[(&str, fn(&String, &String) -> u16)] = &[
//     ("levenshtein", |x: &String, y: &String| {
//         distances::strings::levenshtein(x, y)
//     }),
//     ("needleman-wunsch", |x: &String, y: &String| {
//         distances::strings::nw_distance(x, y)
//     }),
// ];

fn genomic(c: &mut Criterion) {
    // let seed_length = 100;
    // let alphabet = "ACTGN".chars().collect::<Vec<_>>();
    // let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
    // let penalties = distances::strings::Penalties::default();
    // let num_clumps = 16;
    // let clump_size = 16;
    // let clump_radius = 3_u16;
    // let (_, data) = symagen::random_edits::generate_clumped_data(
    //     &seed_string,
    //     penalties,
    //     &alphabet,
    //     num_clumps,
    //     clump_size,
    //     clump_radius,
    // )
    // .into_iter()
    // .unzip::<_, _, Vec<_>, Vec<_>>();

    // let queries = {
    //     let mut indices = (0..data.len()).collect::<Vec<_>>();
    //     indices.shuffle(&mut rand::thread_rng());
    //     indices
    //         .into_iter()
    //         .take(10)
    //         .map(|i| data[i].clone())
    //         .collect::<Vec<_>>()
    // };

    // let seed = Some(42);
    // for &(metric_name, distance_fn) in METRICS {
    //     let metric = Metric::new(distance_fn, true);
    //     let mut dataset = FlatVec::new(data.clone(), metric).unwrap();

    //     let criteria = |c: &Ball<u16>| c.cardinality() > 1;
    //     let (root, _) = Ball::par_new_tree_and_permute(&mut dataset, &criteria, seed);

    //     let mut group = c.benchmark_group(format!("genomic-{}", metric_name));

    //     for radius in [4, 8, 16] {
    //         let id = BenchmarkId::new("RnnClustered", radius);
    //         group.bench_with_input(id, &radius, |b, &radius| {
    //             b.iter_with_large_drop(|| root.batch_search(&dataset, &queries, Algorithm::RnnClustered(radius)));
    //         });

    //         let id = BenchmarkId::new("ParRnnClustered", radius);
    //         group.bench_with_input(id, &radius, |b, &radius| {
    //             b.iter_with_large_drop(|| {
    //                 root.par_batch_par_search(&dataset, &queries, Algorithm::RnnClustered(radius))
    //             });
    //         });
    //     }

    //     for k in [1, 10, 20] {
    //         let id = BenchmarkId::new("KnnRepeatedRnn", k);
    //         group.bench_with_input(id, &k, |b, &k| {
    //             b.iter_with_large_drop(|| root.batch_search(&dataset, &queries, Algorithm::KnnRepeatedRnn(k, 2)));
    //         });

    //         let id = BenchmarkId::new("ParKnnRepeatedRnn", k);
    //         group.bench_with_input(id, &k, |b, &k| {
    //             b.iter_with_large_drop(|| {
    //                 root.par_batch_par_search(&dataset, &queries, Algorithm::KnnRepeatedRnn(k, 2))
    //             });
    //         });

    //         let id = BenchmarkId::new("KnnBreadthFirst", k);
    //         group.bench_with_input(id, &k, |b, &k| {
    //             b.iter_with_large_drop(|| root.batch_search(&dataset, &queries, Algorithm::KnnBreadthFirst(k)));
    //         });

    //         let id = BenchmarkId::new("ParKnnBreadthFirst", k);
    //         group.bench_with_input(id, &k, |b, &k| {
    //             b.iter_with_large_drop(|| root.par_batch_par_search(&dataset, &queries, Algorithm::KnnBreadthFirst(k)));
    //         });

    //         let id = BenchmarkId::new("KnnDepthFirst", k);
    //         group.bench_with_input(id, &k, |b, &k| {
    //             b.iter_with_large_drop(|| root.batch_search(&dataset, &queries, Algorithm::KnnDepthFirst(k)));
    //         });

    //         let id = BenchmarkId::new("ParKnnDepthFirst", k);
    //         group.bench_with_input(id, &k, |b, &k| {
    //             b.iter_with_large_drop(|| root.par_batch_par_search(&dataset, &queries, Algorithm::KnnDepthFirst(k)));
    //         });
    //     }
    // }
    todo!()
}

criterion_group!(benches, genomic);
criterion_main!(benches);
