use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use kdam::{tqdm, BarExt};
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use num_format::{Locale, ToFormattedString};
use serde::{Deserialize, Serialize};

use clam::cluster::PartitionCriteria;
use clam::dataset::{Dataset, VecVec};
use clam::search::cakes::CAKES;

pub mod utils;

use utils::distances;
use utils::search_readers;

fn main() {
    let reports_root = get_reports_root();

    for &(data_name, metric_name) in search_readers::SEARCH_DATASETS {
        // if data_name != "sift" {
        //     continue;
        // }
        if ["deep-image", "nytimes", "lastfm"].contains(&data_name) {
            continue;
        }
        if metric_name == "jaccard" {
            continue;
        }

        let data_dir = {
            let mut path = reports_root.clone();
            path.push(data_name);
            if path.exists() {
                std::fs::remove_dir_all(&path).unwrap();
            }
            std::fs::create_dir(&path).unwrap();
            path
        };

        for &(metric_name, metric) in distances::METRICS {
            let out_dir = {
                let mut path = data_dir.clone();
                path.push(metric_name);
                if path.exists() {
                    std::fs::remove_dir_all(&path).unwrap();
                }
                std::fs::create_dir(&path).unwrap();
                path
            };

            println!();
            println!("Making reports on {data_name} with {metric_name} ...");

            let (data, queries) = search_readers::read_search_data(data_name).unwrap();
            let data = VecVec::new(data, metric, data_name.to_string(), false);

            let car = data.cardinality().to_formatted_string(&Locale::en);
            let dim = data.dimensionality().to_formatted_string(&Locale::en);
            println!("Got data with shape ({car} x {dim}) ...");

            let start = Instant::now();
            let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
            let cakes = CAKES::new(data, Some(42)).build(&criteria);
            let build_time = start.elapsed().as_secs_f32();
            println!("Built CAKES on {data_name} with {metric_name} in {build_time:.3} seconds ...");

            // report_tree(cakes.root(), &out_dir);

            let data = cakes.data();
            let batch_size = 500_000;
            let linear_time = report_linear(data, &queries, &out_dir, batch_size);

            let time = CakesTime {
                data_name,
                metric_name,
                cardinality: data.cardinality(),
                dimensionality: data.dimensionality(),
                build_time,
                num_queries: queries.len(),
                linear_time,
                batch_size,
            };
            let time = serde_json::to_string_pretty(&time).unwrap();
            let time_path = {
                let mut path = out_dir.clone();
                path.push("time-taken.json");
                path
            };
            let mut time_file = std::fs::File::create(&time_path).unwrap();
            time_file.write_all(time.as_bytes()).unwrap();
            println!("Wrote timings file {time_path:?} ...");
        }
    }
}

fn get_reports_root() -> PathBuf {
    let mut path = std::env::current_dir().unwrap();

    path.push("reports");
    assert!(
        path.exists(),
        "Please create a `reports` directory in the root of the clam repo."
    );
    assert!(
        path.is_dir(),
        "Please create a `reports` directory in the root of the clam repo."
    );

    path
}

// fn report_tree(root: &Cluster<f32, f32, VecVec<f32, f32>>, out_dir: &Path) {
//     let clusters = root.subtree();
//     let tree_array = {
//         let names: ArrayRef = {
//             let names = clusters.iter().map(|c| Some(c.name()));
//             Arc::new(StringArray::from_iter(names))
//         };

//         let depths = {
//             let depths = clusters.iter().map(|c| Some(c.depth() as u64));
//             Arc::new(UInt64Array::from_iter(depths))
//         };

//         let (lefts, rights) = {
//             let (lefts, rights): (Vec<_>, Vec<_>) = clusters
//                 .iter()
//                 .map(|c| match c.children() {
//                     Some([left, right]) => (Some(left.name()), Some(right.name())),
//                     None => (None, None),
//                 })
//                 .unzip();
//             (
//                 Arc::new(StringArray::from_iter(lefts)),
//                 Arc::new(StringArray::from_iter(rights)),
//             )
//         };

//         let cardinalities = {
//             let cardinalities = clusters.iter().map(|c| Some(c.cardinality() as u64));
//             Arc::new(UInt64Array::from_iter(cardinalities))
//         };

//         let arg_centers = {
//             let arg_centers = clusters.iter().map(|c| Some(c.arg_center() as u64));
//             Arc::new(UInt64Array::from_iter(arg_centers))
//         };

//         let arg_radii = {
//             let arg_radii = clusters.iter().map(|c| Some(c.arg_radius() as u64));
//             Arc::new(UInt64Array::from_iter(arg_radii))
//         };

//         let radii = {
//             let radii = clusters.iter().map(|c| Some(c.radius()));
//             Arc::new(Float32Array::from_iter(radii))
//         };

//         let lfds = {
//             let lfds = clusters.iter().map(|c| Some(c.lfd() as f32));
//             Arc::new(Float32Array::from_iter(lfds))
//         };

//         let polar_distances = {
//             let polar_distances = clusters.iter().map(|c| c.polar_distance());
//             Arc::new(Float32Array::from_iter(polar_distances))
//         };

//         RecordBatch::try_from_iter_with_nullable(vec![
//             ("name", names, false),
//             ("depth", depths, false),
//             ("left", lefts, true),
//             ("right", rights, true),
//             ("cardinality", cardinalities, false),
//             ("arg_center", arg_centers, false),
//             ("arg_radius", arg_radii, false),
//             ("radius", radii, false),
//             ("lfd", lfds, false),
//             ("polar_distance", polar_distances, true),
//         ])
//         .unwrap()
//     };

//     let tree_file = {
//         let mut path = out_dir.to_path_buf();
//         path.push("tree.arrow");
//         std::fs::File::create(path).unwrap()
//     };
//     let mut writer = FileWriter::try_new(tree_file, tree_array.schema().as_ref()).unwrap();
//     writer.write(&tree_array).unwrap();
//     writer.finish().unwrap();
//     println!("Wrote tree report ...");

//     let leaves = clusters.into_iter().filter(|c| c.is_leaf()).collect::<Vec<_>>();
//     let leaves_array = {
//         let names: ArrayRef = {
//             let names = leaves.iter().map(|c| Some(c.name()));
//             Arc::new(StringArray::from_iter(names))
//         };
//         let indices = {
//             let indices = leaves
//                 .iter()
//                 .map(|c| c.indices().into_iter().map(|i| Some(i as u64)).collect::<Vec<_>>())
//                 .map(Some);
//             Arc::new(LargeListArray::from_iter_primitive::<UInt64Type, _, _>(indices))
//         };

//         RecordBatch::try_from_iter_with_nullable(vec![("name", names, false), ("indices", indices, false)]).unwrap()
//     };
//     let leaves_file = {
//         let mut path = out_dir.to_path_buf();
//         path.push("leaves.arrow");
//         std::fs::File::create(path).unwrap()
//     };
//     let mut writer = FileWriter::try_new(leaves_file, leaves_array.schema().as_ref()).unwrap();
//     writer.write(&leaves_array).unwrap();
//     writer.finish().unwrap();
//     println!("Wrote leaves report ...");
// }

fn report_linear(data: &VecVec<f32, f32>, queries: &[Vec<f32>], out_dir: &Path, batch_size: usize) -> f32 {
    let indices = data.indices();

    let num_batches = {
        let num_batches = indices.len() / batch_size;
        if indices.len() % batch_size == 0 {
            num_batches
        } else {
            num_batches + 1
        }
    };

    let mut time = 0.;
    for (i, batch) in indices.chunks(batch_size).enumerate() {
        let n = i + 1;
        let mut pb = tqdm!(total = queries.len(), desc = format!("Linear Batch {n}/{num_batches}"));
        let mut array = Array2::<f32>::default((0, batch.len()));

        for query in queries.iter() {
            let start = Instant::now();
            let distances = data.query_to_many(query, batch);
            time += start.elapsed().as_secs_f32();

            array.push_row(Array1::from_vec(distances).view()).unwrap();
            pb.update(1);
        }

        let out_path = {
            let mut path = out_dir.to_path_buf();
            path.push(format!("query-distances-batch-{n}-{num_batches}.npy"));
            path
        };
        write_npy(&out_path, &array).unwrap();
    }
    time
}

#[derive(Debug, Serialize, Deserialize)]
struct CakesTime<'a> {
    data_name: &'a str,
    metric_name: &'a str,
    cardinality: usize,
    dimensionality: usize,
    build_time: f32,
    num_queries: usize,
    linear_time: f32,
    batch_size: usize,
}
