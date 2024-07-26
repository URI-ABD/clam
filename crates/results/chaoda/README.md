# Experiments with CHAODA

This is a binary for running experiments with CHAODA to help drive development in CLAM.

## Usage

The expected input parameters are:

1. The path to the data directory.
2. Whether to use the pre-trained model or not.

Run the following command from the workspace root to run the experiments:

```bash
cargo run -r -p results-chaoda -- /path/to/data false
```
