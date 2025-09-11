# CLAM Shell

## Quickstart

```bash
# 1. Create an ignored directory to experiment with.
mkdir -p target/experiments
# 2. Generate data with 90/10 split
cargo run --package shell -- \
    generate-data generate \
        --num-vectors 50 \
        --dimensions 10 \
        --filename target/experiments/data \
        --data-type f32 \
        --partitions 90,10
# 3. Build the tree
cargo run --package shell -- \
    --inp-path target/experiments/data-45.npy \
    cakes build \
        --out-dir target/experiments/
# 4. Search for some queries.
cargo run --package shell -- \
    -i target/experiments/data-45.npy cakes search \
        --tree-path ./target/experiments/tree.bin \
        --instances-path ./target/experiments/data-5.npy \
        --query-algorithms knn-linear:k=2 \
        --query-algorithms knn-linear:k=5 \
        --output-path ./target/experiments/results.json
```
