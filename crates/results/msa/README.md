# Results for MSA with CLAM

## Usage

If you want to run the MSA on all a sequences in a fasta file, you can use the following command:

```shell
cargo run -r -p results-msa -- \
    -i ../data/string-data/greengenes/gg_13_5.fasta \
    -o ../data/string-data/greengenes/msa-results
```

If you want to run the MSA on a subset of the sequences in a fasta file, you can use use the optional `-n` flag to specify the number of sequences to use:

```shell
cargo run -r -p results-msa -- \
    -i ../data/string-data/greengenes/gg_13_5.fasta \
    -n 1000 \
    -o ../data/string-data/greengenes/msa-results
```

## Citation

TODO...
