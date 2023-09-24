#!/bin/bash

# Requires:
# - Rust version 1.72.0 or higher: https://www.rust-lang.org/tools/install
# - Ann Datasets: https://github.com/erikbern/ann-benchmarks#data-sets

# Set input and output directories
input_dir="../data/search_small/as_npy"
output_dir="../data/search_small/reports"

echo "Starting ann-benchmarks at: $(date)"

# Compile cakes-results
cargo build --release --all-targets

for dataset in "deep-image" "fashion-mnist" "gist" "glove-25" "glove-50" "glove-100" "glove-200" "mnist" "sift"
do
    for k in 10 100
    do
        ./target/release/cakes-results \
            --input-dir $input_dir \
            --output-dir $output_dir \
            --dataset $dataset \
            --k $k
    done
done
