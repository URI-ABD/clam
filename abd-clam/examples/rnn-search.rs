use std::{fs::File, io::prelude::*, time::Instant};

use num_format::{Buffer, CustomFormat, Locale, ToFormattedString};

use abd_clam::{rnn, Cakes, PartitionCriteria, VecDataset, COMMON_METRICS_F32};

pub mod utils;

use utils::anomaly_readers;

fn main() -> std::io::Result<()> {
    let mut lines = vec![
        "data_name,cardinality,dimensionality,metric_name,build_time(micro_sec),factor,throughput(qps),mean_result_size"
            .to_string(),
    ];

    // for &(data_name, metric_name) in search_readers::SEARCH_DATASETS {
    for &data_name in anomaly_readers::ANOMALY_DATASETS {
        // if metric_name == "jaccard" {
        //     continue;
        // }
        let mut report = search(Some(42), data_name);
        println!("{report}", report = report.join("\n"));
        lines.append(&mut report);
    }

    let lines = lines.join("\n");
    let mut out_file = File::create("rnn-report.csv")?;
    out_file.write_all(lines.as_bytes())?;

    Ok(())
}

fn search(seed: Option<u64>, data_name: &str) -> Vec<String> {
    let mut report = vec![];

    // let (data, queries) = search_readers::read_search_data(data_name).unwrap();
    let (data, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();
    let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();

    // let queries = queries.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let num_queries = 10_000;
    let queries = (0..num_queries)
        .map(|v| v % data.len())
        .map(|i| data[i])
        .collect::<Vec<_>>();

    let line_start = vec![
        data_name.to_string(),
        usize_to_string(data.len()),
        usize_to_string(data[0].len()),
    ];

    for &(metric_name, metric) in &COMMON_METRICS_F32[..3] {
        let mut line_metric = line_start.clone();
        line_metric.push(metric_name.to_string());

        let name = format!("{data_name}-{metric_name}");
        let data = VecDataset::new(name, data.clone(), metric, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        let start = Instant::now();
        let cakes = Cakes::new(data, seed, criteria);
        line_metric.push(f32_to_string(start.elapsed().as_micros() as f32));

        let num_queries = queries.len();

        let factors = [100_u32, 50, 25, 10].as_slice();

        for &factor in factors {
            let mut line_factor = line_metric.clone();
            line_factor.push(factor.to_formatted_string(&Locale::en));

            let radius = cakes.radius() / factor as f32;

            let start = Instant::now();
            let results = cakes.batch_rnn_search(&queries, radius, rnn::Algorithm::Clustered);
            let duration = start.elapsed().as_secs_f32();

            let num_results = results.iter().map(Vec::len).collect::<Vec<_>>();
            let mean_result_size = num_results.iter().sum::<usize>() as f32 / num_queries as f32;

            line_factor.push(f32_to_string(num_queries as f32 / duration));
            line_factor.push(f32_to_string(mean_result_size));

            report.push(line_factor.join(","));

            if mean_result_size >= 500. {
                break;
            }
        }
    }

    report
}

fn f32_to_string(f: f32) -> String {
    let format = CustomFormat::builder().separator("_").build().unwrap();

    let mut trunc = Buffer::new();
    trunc.write_formatted(&(f.trunc() as u32), &format);

    let mut fract = Buffer::new();
    fract.write_formatted(&((f.fract() * 100.0).trunc() as u32), &format);

    trunc.to_string() + "." + &fract
}

fn usize_to_string(u: usize) -> String {
    let format = CustomFormat::builder().separator("_").build().unwrap();

    let mut trunc = Buffer::new();
    trunc.write_formatted(&u, &format);

    trunc.to_string()
}
