#!/bin/bash

# Requires:
# - Rust version 1.72.0 or higher: https://www.rust-lang.org/tools/install
# - Ann Datasets: https://github.com/erikbern/ann-benchmarks#data-sets

# Set input and output directories
input_dir="../data/ann-benchmarks/datasets"
output_dir="../data/ann-benchmarks/scaling_reports"

echo "Starting scaling-benchmarks at: $(date)"

# Compile cakes-results
cargo build --release --bin scaling-results

for dataset in "fashion-mnist" "glove-25" "glove-100" "sift" "random-128-euclidean" "gist" "deep-image"
do
    for error_rate in 0.01 0.05 0.1
    do
        ./target/release/scaling-results \
            --input-dir $input_dir \
            --output-dir $output_dir \
            --dataset $dataset \
            --error-rate $error_rate \
            --ks 10 100
    done
done
