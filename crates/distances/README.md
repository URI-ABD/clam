# Distances (v1.6.3)

Fast and generic distance functions for high-dimensional data.

## Usage

Add this to your project:

```shell
> cargo add distances@1.6.3
```

Use it in your project:

```rust
use distances::Number;
use distances::vectors::euclidean;

let a = [1.0_f32, 2.0, 3.0];
let b = [4.0_f32, 5.0, 6.0];

let distance: f32 = euclidean(&a, &b);

assert!((distance - (27.0_f32).sqrt()).abs() < 1e-6);
```

## Features

- [x] A `Number` trait to abstract over different numeric types.
  - [x] Distance functions are generic over the return type implementing `Number`.
  - [x] Distance functions may also be generic over the input type being a collection of `Number`s.
- [ ] SIMD accelerated implementations for float types.
- [ ] Python bindings with `maturin` and `pyo3`.
- [ ] `no_std` support.

## Available Distance Functions

- [ ] Vectors (high-dimensional data):
  - [x] `euclidean`
  - [x] `squared_euclidean`
  - [x] `manhattan`
  - [x] `chebyshev`
  - [x] `minkowski`
    - General Lp-norm.
  - [x] `minkowski_p`
    - General Lp-norm to the `p`th power.
  - [x] `cosine`
  - [x] `hamming`
  - [x] `canberra`
    - [Canberra Distance](https://en.wikipedia.org/wiki/Canberra_distance)
  - [x] `bray_curtis`
    - [Bray-Curtis Distance](https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity)
  - [ ] `pearson`
    - `1.0 - r` where `r` is the [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [ ] Probability distributions:
  - [ ] `wasserstein`
    - [Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric)
  - [ ] `bhattacharyya`
    - [Bhattacharyya Distance](https://en.wikipedia.org/wiki/Bhattacharyya_distance)
  - [ ] `hellinger`
    - [Hellinger Distance](https://en.wikipedia.org/wiki/Hellinger_distance)
- [ ] String data, e.g. for genomic sequences:
  - [x] `levenshtein`
  - [x] `needleman_wunsch`
  - [ ] `smith_waterman`
  - [x] `hamming`
  - [ ] Normalized versions of the above.
- [ ] Sets:
  - [x] `jaccard`
  - [ ] `hausdorff`
    - [Hausdorff Distance](https://en.wikipedia.org/wiki/Hausdorff_distance)
- [ ] Graphs:
  - [ ] `tanamoto`
- [ ] Time series:
  - [ ] `dtw`
    - [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping)
  - [ ] `msm`
    - [Move-Split-Merge](https://doi.org/10.1109/TKDE.2012.88)
  - [ ] `erp`
    - [Edit distance with Real Penalty](https://rdrr.io/cran/TSdist/man/ERPDistance.html)

## Contributing

Contributions are welcome, encouraged, and appreciated!
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the [MIT license](LICENSE-MIT).
