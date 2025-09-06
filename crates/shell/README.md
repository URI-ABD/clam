# CLAM Shell

## Usage

Run the following command to see the available options and subcommands:

```bash
cargo run --release --package shell -- --help
```

So far, the available subcommands are as follows.

### 1. Generate Data

This will create synthetic datasets for testing and getting to know the CLI.

```bash
cargo run -p shell -- generate-data --help
```

### 2. CAKES

This subcommand allows building and searching various tree-based data structures for fast nearest neighbor search.

```bash
cargo run -p shell -- cakes --help

cargo run -p shell -- cakes build --help

cargo run -p shell -- cakes search --help
```

### 3. MuSAlS

This subcommand allows building and evaluating multiple sequence alignments (MSAs) using the MuSAlS algorithm.

```bash
cargo run -p shell -- musals --help

cargo run -p shell -- musals build --help

cargo run -p shell -- musals evaluate --help
```

## Examples

### 1. Generate Data

```bash
# Create an ignored directory to experiment with.
mkdir -p target/experiments

# Generate 50 small random vectors of dimension 10, partitioned into 90% train and 10% test sets.
cargo run --release --package shell -- \
    --out-path target/experiments/data/small-vectors.npy \
    --seed 42 \
    --log-name gen-data.log \
    generate-data generate \
        --num-vectors 50 \
        --dimensions 10 \
        --data-type f32 \
        --partitions 90,10 \
        --min-val 0.0 \
        --max-val 1.0
```

### 2. CAKES

We use the data generated above to build a tree and perform some searches.

```bash
# Build the tree
cargo run --release --package shell -- \
    --inp-path target/experiments/data/small-vectors-45.npy \
    --out-path target/experiments/small-trees \
    --metric euclidean \
    --log-name cakes-build.log \
    cakes build

# Search for some queries.
cargo run --release --package shell -- \
    --inp-path target/experiments/small-trees \
    --out-path target/experiments/small-results.json \
    --log-name cakes-search.log \
    cakes search \
        --queries-path ./target/experiments/data/small-vectors-5.npy \
        --cakes-algorithms knn-linear:k=2 \
        --cakes-algorithms knn-linear:k=5
```

### 3. MuSAlS

We assume you have a FASTA file at `target/experiments/sequences.fasta` of protein sequences to align.

```bash
# Build the MSA
cargo run -rp shell -- \
    -i target/experiments/sequences.fasta \
    -o target/experiments/msa-results \
    -m levenshtein \
    musals -c blosum62 \
    build

# Evaluate the quality of the MSA on 1000 random sequences from the input FASTA file.
cargo run -rp shell -- \
    -i target/experiments/msa-results \
    -o target/experiments/msa-results/msa-eval.json \
    -m levenshtein \
    -n 1000 \
    musals -c blosum62 \
    evaluate \
    -q distance-distortion \
    -q gap-fraction \
    -q mismatch-fraction \
    -q sum-of-pairs
```

We could also subsample the input sequences before building the MSA tree to experiment with different cost matrices. The `-f` flag for `build` will write the
aligned sequences to a FASTA file, in addition to the binary tree files.

```bash
# Build the MSA with 1000 random sequences from the input FASTA file.
cargo run -rp shell -- \
    -i target/experiments/sequences.fasta \
    -o target/experiments/msa-results-1k \
    -m levenshtein \
    -n 1000 \
    musals -c blosum62 \
    build -f
```

cargo run -rp shell -- \
    -i ../data/string-data/greengenes/msa-results/gg-12 \
    -o ../data/string-data/greengenes/msa-results/gg-12/msa-eval.json \
    -m levenshtein \
    -n 1000 \
    musals -c blosum62 \
    evaluate \
    -q distance-distortion \
    -q gap-fraction \
    -q mismatch-fraction \
    -q sum-of-pairs
