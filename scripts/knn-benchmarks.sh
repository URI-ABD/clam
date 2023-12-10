#!/bin/bash

# Requires:
# - Rust version 1.72.0 or higher: https://www.rust-lang.org/tools/install
# - Ann Datasets: https://github.com/erikbern/ann-benchmarks#data-sets

# Set input and output directories
input_dir="../data/ann-benchmarks/datasets"
output_dir="../data/ann-benchmarks/knn_reports"

echo "Starting ann-benchmarks at: $(date)"

# Compile cakes-results
cargo build --release --bin knn-results

# for dataset in "deep-image" "fashion-mnist" "gist" "glove-25" "glove-100" "mnist" "sift"
for dataset in "fashion-mnist"
do
    ./target/release/knn-results \
        --input-dir $input_dir \
        --output-dir $output_dir \
        --dataset $dataset \
        --ks 10 100
done
