# Benchmarks for CAKES Search Algorithms

This is crate provides a CLI to run benchmarks for the CAKES search algorithms and reproduce the results from our paper.

## Reproducing the Results

Let's say you have data from the [ANN-Benchmarks suite](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file#data-sets) in a directory `../data/input` and you want to run the benchmarks for the CAKES search algorithms on the `sift` dataset.
You can run the following command:

```bash
cargo run --release --package bench-cakes -- \
    --inp-dir ../data/input/ \
    --dataset sift \
    --out-dir ../data/output/ \
    --seed 42 \
    --num-queries 10000 \
    --max-power 7 \
    --max-time 300 \
    --balanced-data \
    --permuted-trees
```

This will run the CAKES search algorithms on the `sift` dataset with 10000 search queries.
The results will be saved in the directory `../data/output/`.
The dataset will be augmented by powers of 2 up to 2^7.
Each algorithm will be run for at least 300 seconds.
The `--balanced` flag will build trees with balanced partitions.
The `--permuted` flag will permute the dataset into depth-first order after building the tree.

There are several other available options.
Running the following command will provide documentation on how to use the CLI:

```bash
cargo run --release --package bench-cakes -- --help
```

## Plotting the Results

The outputs from the benchmarks can be plotted using the python package we provide at `../py-cakes`.
See the associated README for more information.
