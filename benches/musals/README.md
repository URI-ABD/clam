# Benchmarks for CLAM-MUSALS

This is crate provides a CLI to run benchmarks for the MUSALS algorithm for building multiple sequence alignments (MSAs) on biological sequences.

## Usage

Run the following command to see the usage information:

```shell
cargo run -rp bench-musals -- --help
```

If you want to build the MSA on all a sequences in a fasta file, you can use the following command:

```shell
cargo run -rp bench-musals -- \
    -i ../data/string-data/greengenes/gg_13_5.fasta \
    -o ../data/string-data/greengenes/musals-results \
    -m extended-iupac
```

If you want to build the MSA on a subset of the sequences in a fasta file, you can use use the optional `-n` flag to specify the number of sequences to use:

```shell
cargo run -rp bench-musals -- \
    -i ../data/string-data/greengenes/gg_13_5.fasta \
    -o ../data/string-data/greengenes/musals-results \
    -m extended-iupac \
    -n 1000
```

## Citation

TODO...
