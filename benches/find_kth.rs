// pub mod utils;
// use criterion::criterion_group;
// use criterion::criterion_main;
// use criterion::BenchmarkId;
// use criterion::Criterion;

// use rand::Rng;
// use std::cmp::Ordering;

// use clam::prelude::*;
// use clam::search::Delta;
// use clam::search::Grain;
// use clam::search::KnnSieve;
// use clam::Tabular;

// fn sort_and_index<T: clam::Number>(v: &mut [Grain<T>], k: usize) -> f64 {
//     v.sort_by(|a, b| a.ord_by_d_max(b));
//     println!("grains at current state sort");
//     for g in v.iter() {
//         println!("{:?}", g.d_max);
//     }
//     println!();
//     v[k].d_max
// }

// fn unstable_sort_and_index<T: clam::Number>(v: &mut [Grain<T>], k: usize) -> f64 {
//     v.sort_unstable_by(|a, b| a.ord_by_d_max(b));
//     v[k].d_max
// }

// fn _find_kth<T: clam::Number>(
//     grains: &mut [Grain<T>],
//     cumulative_cardinalities: &mut [usize],
//     k: usize,
//     l: usize,
//     r: usize,
//     delta: &Delta,
// ) -> f64 {
//     println!();
//     println!("grains at current state kth");
//     for g in grains.iter() {
//         println!("{:?}", g.d_max);
//     }
//     println!();

//     let mut cardinalities = (0..1)
//         .chain(cumulative_cardinalities.iter().copied())
//         .zip(cumulative_cardinalities.iter().copied())
//         .map(|(prev, next)| next - prev)
//         .collect::<Vec<_>>();

//     assert_eq!(cardinalities.len(), grains.len());
//     println!("lens: {}", grains.len());

//     let position = partition(grains, &mut cardinalities, l, r, delta);
//     println!("pos: {position}");

//     let mut cumulative_cardinalities = cardinalities
//         .iter()
//         .scan(0, |acc, v| {
//             *acc += *v;
//             Some(*acc)
//         })
//         .collect::<Vec<_>>();
//     println!("ccs: {cumulative_cardinalities:?}");

//     match cumulative_cardinalities[position].cmp(&k) {
//         Ordering::Less => {
//             println!("too small: {}-{}", position + 1, r);
//             _find_kth(grains, &mut cumulative_cardinalities, k, position + 1, r, delta)
//         }
//         Ordering::Equal => match delta {
//             Delta::D => grains[position].d,
//             Delta::Max => grains[position].d_max,
//             Delta::Min => grains[position].d_min,
//         },
//         Ordering::Greater => {
//             if (position > 0) && (cumulative_cardinalities[position - 1] > k) {
//                 println!("too big: {}-{}", l, position - 1);
//                 _find_kth(grains, &mut cumulative_cardinalities, k, l, position - 1, delta)
//             } else {
//                 println!("im here");
//                 match delta {
//                     Delta::D => {
//                         println!("im delta");
//                         grains[position].d
//                     }
//                     Delta::Max => {
//                         println!("im delta max");
//                         grains[position].d_max
//                     }
//                     Delta::Min => {
//                         println!("im delta min");
//                         grains[position].d_min
//                     }
//                 }
//             }
//         }
//     }
// }

// fn partition<T: clam::Number>(
//     grains: &mut [Grain<T>],
//     cardinalities: &mut [usize],
//     l: usize,
//     r: usize,
//     delta: &Delta,
// ) -> usize {
//     let pivot = (l + r) / 2; // Check for overflow
//     swaps(grains, cardinalities, pivot, r);

//     let (mut a, mut b) = (l, l);
//     while b < r {
//         let grain_comp = match delta {
//             Delta::D => (grains[b]).ord_by_d(&grains[r]),
//             Delta::Max => (grains[b]).ord_by_d_max(&grains[r]),
//             Delta::Min => (grains[b]).ord_by_d_min(&grains[r]),
//         };

//         if grain_comp == Ordering::Less {
//             swaps(grains, cardinalities, a, b);
//             a += 1;
//         }
//         b += 1;
//     }

//     swaps(grains, cardinalities, a, r);

//     a
// }

// pub fn swaps<T>(grains: &mut [T], cardinalities: &mut [usize], i: usize, j: usize) {
//     grains.swap(i, j);
//     cardinalities.swap(i, j);
// }

// fn find_kth(c: &mut Criterion) {
//     let mut group = c.benchmark_group("find_kth");
//     group.significance_level(0.05).sample_size(10);

//     let mut rng = rand::thread_rng();
//     let data: Vec<Vec<f64>> = (0..300)
//         .map(|_| (0..5).map(|_| rng.gen_range(0.0..10.0)).collect())
//         .collect();
//     let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
//     let metric = metric_from_name::<f64>("euclideansq", false).unwrap();
//     let space = clam::TabularSpace::new(&dataset, metric.as_ref());

//     let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
//     let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
//     let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);
//     let flat_tree = cluster.subtree();

//     let query_ind = rng.gen_range(0..data.len());
//     //let k = rng.gen_range(0..5);
//     let k = 3;

//     let sieve = KnnSieve::new(flat_tree.clone(), &data[query_ind], k);

//     let diffs: Vec<usize> = ((k + 1)..flat_tree.len() - 2).step_by(10).collect();

//     for &diff in diffs.iter() {
//         let kth_from_find_kth = _find_kth(
//             &mut sieve.grains.clone()[0..(diff + 1)],
//             &mut sieve.cumulative_cardinalities.clone()[0..(diff + 1)],
//             sieve.k,
//             0,
//             diff,
//             &Delta::Max,
//         );
//         println!("kth from find kth: {}", &kth_from_find_kth);
//         // group.bench_with_input(BenchmarkId::new("find_kth", diff), &diff, |b, &diff| {
//         //     b.iter(|| {
//         //         _find_kth(
//         //             &mut sieve.grains.clone()[0..(diff + 1)],
//         //             &mut sieve.cumulative_cardinalities.clone()[0..(diff + 1)],
//         //             sieve.k,
//         //             0,
//         //             diff,
//         //             &Delta::Max,
//         //         )
//         //     });
//         // });

//         let kth_from_sort = sort_and_index(&mut sieve.grains.clone()[0..(diff + 1)], sieve.k);
//         println!("kth from sort: {}", &kth_from_sort);

//         // group.bench_with_input(BenchmarkId::new("sort", diff), &diff, |b, &_diff| {
//         //     b.iter(|| sort_and_index(&mut sieve.grains.clone()[0..(diff + 1)], sieve.k));
//         // });

//         group.bench_with_input(BenchmarkId::new("unstable_sort", diff), &diff, |b, &_diff| {
//             b.iter(|| unstable_sort_and_index(&mut sieve.grains.clone()[0..(diff + 1)], sieve.k));
//         });

//         // assert_eq!(kth_from_find_kth, kth_from_sort);
//     }
//     group.finish();
// }

// criterion_group!(benches, find_kth);
// criterion_main!(benches);
