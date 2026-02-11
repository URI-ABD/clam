//! Benchmarks for CAKES

#![expect(missing_docs, unused)]

use std::usize;

use abd_clam::{
    DistanceValue, PartitionStrategy, Tree,
    cakes::{self, ParSearch, approximate, search_quality_stats},
    partition_strategy::{BranchingFactor, SpanReductionFactor},
};
use rand::prelude::*;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::prelude::*;

mod utils;

use utils::ann_benchmarks::{AnnDataset, base_dir};

fn bench_for_args<Id, I, T, A, M>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tree: &Tree<Id, I, T, A, M>,
    queries: &[I],
    multiplier: usize,
    ks: &[usize],
    rs: &[T],
) where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    for &k in ks {
        let true_hits = cakes::KnnLinear(k).par_batch_search(tree, queries);

        // Benchmark the exact algorithms
        bench_one_alg(group, tree, &cakes::KnnDfs(k), queries, &true_hits, multiplier);
        bench_one_alg(group, tree, &cakes::KnnBfs(k), queries, &true_hits, multiplier);
        bench_one_alg(group, tree, &cakes::KnnRrnn(k), queries, &true_hits, multiplier);

        // Benchmark the approximate algorithms
        for n in [10, 100, 1000] {
            // Varying number of leaves explored
            let l_alg = approximate::KnnDfs::new(k, n, usize::MAX);
            bench_one_alg(group, tree, &l_alg, queries, &true_hits, multiplier);
            // Varying number of distance computations performed
            let d_alg = approximate::KnnDfs::new(k, usize::MAX, n * 100);
            bench_one_alg(group, tree, &d_alg, queries, &true_hits, multiplier);
        }
    }

    for &r in rs {
        let true_hits = cakes::RnnLinear(r).par_batch_search(tree, queries);

        // Benchmark the exact algorithms
        bench_one_alg(group, tree, &cakes::RnnChess(r), queries, &true_hits, multiplier);
    }
}

fn bench_one_alg<Id, I, T, A, M, Alg>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tree: &Tree<Id, I, T, A, M>,
    alg: &Alg,
    queries: &[I],
    true_hits: &[Vec<(usize, T)>],
    multiplier: usize,
) where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Alg: ParSearch<Id, I, T, A, M>,
{
    let id = BenchmarkId::new(alg.name(), multiplier);
    group.bench_function(id, |b| b.iter_with_large_drop(|| alg.par_batch_search(&tree, &queries)));

    let all_clusters = tree.cluster_map().values().collect::<Vec<_>>();
    let size_of_tree = all_clusters.len();
    let max_depth = all_clusters.iter().map(|c| c.depth()).max().unwrap_or(0);

    let all_leaves = all_clusters.iter().filter(|c| c.is_leaf()).copied().collect::<Vec<_>>();
    let leaf_fraction = all_leaves.len() as f64 / size_of_tree as f64;
    let mean_leaf_cardinality = all_leaves.iter().map(|c| c.cardinality()).sum::<usize>() as f64 / all_leaves.len() as f64;
    let singleton_fraction = all_leaves.iter().filter(|c| c.is_singleton()).count() as f64 / all_leaves.len() as f64;

    println!("Tree stats for dataset with cardinality {} after multiplier {multiplier}:", tree.cardinality());
    println!(
        "    Number of clusters: {size_of_tree}, Ratio: {:.8}",
        size_of_tree as f64 / tree.cardinality() as f64
    );
    println!("    Max depth: {max_depth}");
    println!("    Leaf fraction of clusters: {leaf_fraction:.8}, mean leaf cardinality: {mean_leaf_cardinality:.8}");
    println!("    Singleton fraction of leaves: {singleton_fraction:.8}");

    let pred_hits = alg.par_batch_search(&tree, queries);
    let recall_stats = search_quality_stats(true_hits, &pred_hits);
    println!("Search quality of {}:", alg.name());
    for (stat_name, stat_value) in recall_stats {
        println!("    {stat_name}: {stat_value:.8}");
    }
}

/// Run benchmarks for a given dataset, doubling the dataset size each iteration until `max_items` is reached.
fn run_group<P: AsRef<std::path::Path>, R: rand::Rng>(
    c: &mut Criterion,
    rng: &mut R,
    dataset: &AnnDataset,
    base: &P,
    max_items: usize,
    max_queries: usize,
    shuffle: bool,
    branching_factors: &[usize],
    ks: &[usize],
    rs: &[f32],
) {
    let mut items = dataset.read_train(base, if shuffle { Some(rng) } else { None }).unwrap();
    let metric = dataset.metric();
    let queries = dataset.read_test(base, if shuffle { Some(rng) } else { None }).unwrap();
    let queries = queries[..(max_queries.min(queries.len()))].to_vec();

    // We will change this error value at the end of each iteration
    let mut augmentation_error = 0.0;
    for multiplier in (0..).map(|p| 2_usize.pow(p)) {
        if multiplier > 1 && items.len() * multiplier > max_items {
            // Stop if augmentation would exceed max_items
            break;
        }

        if multiplier > 1 {
            // We have had at least one iteration, so we augment the data with small random noise
            let dimensionality = items[0].len();
            let dimensional_error = augmentation_error / (dimensionality as f32).sqrt();
            let perturbations = symagen::random_data::random_tabular(items.len(), dimensionality, 1.0 - dimensional_error, 1.0 + dimensional_error, rng);
            items = items
                .into_par_iter()
                .zip(perturbations)
                .flat_map(|(point, perturbation)| {
                    let perturbed_point = point.iter().zip(perturbation).map(|(&x, y)| x + y).collect();
                    [point, perturbed_point]
                })
                .collect();
        }

        // TODO(Najib): Add more strategies back in once we have a better sense of which ones are promising.
        // let strategies = {
        //     let mut strategies: Vec<PartitionStrategy<_>> = vec![];

        //     for &bf in branching_factors {
        //         strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Fixed(bf)));
        //     }
        //     for &bf in branching_factors {
        //         if bf > 2 {
        //             strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Adaptive(bf)));
        //         }
        //     }
        //     strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Logarithmic));

        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Sqrt2));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Two));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::E));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Pi));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Phi));

        //     strategies
        // };

        let strategies = vec![PartitionStrategy::default()]; // For now, only use the default strategy for benchmarks

        for strategy in &strategies {
            let mut group = c.benchmark_group(format!("CAKES-{}-{strategy}", dataset.name()));
            config_group(&mut group, queries.len());

            // Build a tree with no annotations and benchmark the search algorithms
            if shuffle {
                items.shuffle(rng);
            }
            let id_items = items.into_iter().enumerate().collect::<Vec<_>>();

            println!("Building Tree");
            let tree_start = std::time::Instant::now();
            let tree = Tree::par_new(id_items, metric, strategy, &|_| ()).unwrap();
            let tree_time = tree_start.elapsed();
            println!("Built Tree in {:.6}", tree_time.as_secs_f32());

            bench_for_args(&mut group, &tree, &queries, multiplier, ks, rs);

            // Set the augmentation error to 0.1% of the radius of the root ball for the next augmentation
            augmentation_error = tree.root().radius() / 1000.0;
            items = tree.take_items().into_iter().map(|(_, p)| p).collect();

            group.finish();
        }
    }
}

fn config_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n_queries: usize) {
    group.sample_size(10);
    group.throughput(criterion::Throughput::Elements(n_queries as u64));

    let plot_config = criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    group.plot_config(plot_config);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let datasets = [
        // For the paper
        AnnDataset::FashionMnist,
        AnnDataset::Glove25,
        AnnDataset::Sift,
        // The rest
        AnnDataset::Mnist,
        AnnDataset::Gist,
        AnnDataset::Glove50,
        AnnDataset::Glove100,
        AnnDataset::Glove200,
        AnnDataset::DeepImage,
    ];

    // Whether to shuffle the dataset before building the tree
    let shuffle = false;
    let mut rng = rand::rng();

    // Set these parameters to control the runtime of the benchmarks. These settings go all out and will take a long time.
    let max_items = 100_000;
    let max_queries = 100;
    let branching_factors = [2];
    let ks = [10, 100];
    let rs = [0.1, 1.0, 10.0, 100.0, 1000.0];

    let base = base_dir().unwrap();
    for dataset in &datasets {
        if !matches!(dataset, AnnDataset::FashionMnist) {
            continue; // Targeting dataset for hyperparameter tuning
        }
        // For the paper, only use the first 3 datasets
        run_group(c, &mut rng, dataset, &base, max_items, max_queries, shuffle, &branching_factors, &ks, &rs);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
