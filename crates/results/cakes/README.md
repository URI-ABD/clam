# Results for Search and Compression with CAKES

## Usage

```bash
cargo run --release --bin results-cakes -- --help
```

The arguments are as follows:

- `--dataset` or `-d`: Name of the dataset. One of:
  - `gg_13_5`: Greengenes 13.5
  - `gg_12_10`: Greengenes 12.10
  - `gg_12_10_aligned`: Greengenes 12.10 pre-aligned
- `--metric` or `-m`: Name of the distance function to use. One of:
  - `lev`: Levenshtein distance
  - `nw`: Needleman-Wunsch distance
  - `ham`: Hamming distance
- `--inp-dir` or `-i`: Path to the input directory.
- `--out-dir` or `-o`: Path to the output directory.
