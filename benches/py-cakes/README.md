# Plotting results of the CAKES benchmarks

This package provides a CLI for plotting the results of the CAKES benchmarks.

You mush first run any benchmarks you want to plot using the `bench-cakes` crate we provide at `../cakes`.
See the associated README for more information.

## Usage

Create a virtual environment with Python 3.9 or later and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

Install the package with the following command:

```bash
python -m pip install -e benches/py-cakes
```

Let's say you ran benchmarks for CAKES and saved results in a directory `../data/output`.
You now want to generate the plots and save them in a directory `../data/summary`.
You can do this with the following command:

```bash
python -m py_cakes summarize-rust --inp-dir ../data/output --out-dir ../data/summary
```
