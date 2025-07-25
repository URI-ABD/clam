# Algorithms for Big Data: Distances (v1.0.4)

This package contains algorithms for computing distances between data points.
It is a thin Python wrapper around the `distances` crate, in Rust.
It provides drop-in replacements for the distance functions in `scipy.spatial.distance`.

## Supported Distance Functions

## Installation

```bash
pip install abd-distances
```

## Usage

```python
import math

import numpy
import abd_distances.simd as distance

a = numpy.array([i for i in range(10_000)], dtype=numpy.float32)
b = a + 1.0

dist = distance.euclidean(a, b)

assert math.fabs(dist - 100.0) < 1e-6

print(dist)
# 100.0
```

### Vector Distances

- [x] Bray-Curtis: `abd_distances.vector.braycurtis`
- [x] Canberra: `abd_distances.vector.canberra`
- [x] Chebyshev: `abd_distances.vector.chebyshev`
- [ ] Correlation
- [x] Cosine: `abd_distances.vector.cosine`
- [x] Euclidean: `abd_distances.vector.euclidean`
- [ ] Jensen-Shannon
- [ ] Mahalanobis
- [x] Manhattan: `abd_distances.vector.manhattan` and `abd_distances.vector.cityblock`
- [x] Minkowski: `abd_distances.vector.minkowski`
- [ ] Standardized Euclidean
- [x] Squared Euclidean: `abd_distances.vector.sqeuclidean`
- [x] Pairwise Distances: `abd_distances.vector.cdist` and `abd_distances.vector.pdist`
- [ ] ...

#### Boolean Distances

- [ ] Dice
- [ ] Hamming
- [ ] Jaccard
- [ ] Kulczynski 1D
- [ ] Rogers-Tanimoto
- [ ] Russell-Rao
- [ ] Sokal-Michener
- [ ] Sokal-Sneath
- [ ] Yule
- [ ] ...

### SIMD-Accelerated Vector Distances

- [x] Euclidean: `abd_distances.simd.euclidean`
- [x] Squared Euclidean: `abd_distances.simd.sqeuclidean`
- [x] Cosine: `abd_distances.simd.cosine`
- [x] Pairwise Distances: `abd_distances.simd.cdist` and `abd_distances.simd.pdist`
- [ ] ...

### String Distances

- [x] Hamming: `abd_distances.strings.hamming`
- [x] Levenshtein: `abd_distances.strings.levenshtein`
- [x] Needleman-Wunsch: `abd_distances.strings.needleman_wunsch`
- [ ] Smith-Waterman
- [ ] Pairwise Distances
- [ ] ...

## Benchmarks

To reproduce benchmarks, clone the repo and run the following:

```shell
cargo build --release --workspace
uv sync --all-packages
uv run richbench --markdown pypi/distances/benches
```

To reproduce the plots,

```shell
cd pypi/distances
python -m plots
```

### SIMD-Accelerated Vector Distance Benchmarks

These benchmarks were run on an Intel Core i7-11700KF CPU @ 4.900GHz, using **a single thread**.
The OS was Arch Linux, with kernel version 6.7.4-arch1-1.

The "Min", "Max", and "Mean" columns show the minimum, maximum, and mean times (in seconds), respectively, taken to compute the pairwise distances using the functions from `scipy.spatial.distance`.
The "Min (+)", "Max (+)", and "Mean (+)" columns show the speedup of the this package's functions over the `scipy` functions.
All pairwise distances (`cdist` and `pdist`) were computed for 200x200 vectors of 500 dimensions, and the average time was taken over 100 runs.
All individual distances were computed for 20x20 vectors of 500 dimensions, and the average time was taken over 100 runs.

#### 32-bit floats

|          Benchmark | Min     | Max     | Mean    | Min (+)         | Max (+)         | Mean (+)        |
|--------------------|---------|---------|---------|-----------------|-----------------|-----------------|
|   cdist, euclidean | 2.146   | 2.237   | 2.174   | 0.163 (13.2x)   | 0.232 (9.7x)    | 0.189 (11.5x)   |
| cdist, sqeuclidean | 2.123   | 2.257   | 2.156   | 0.154 (13.7x)   | 0.189 (12.0x)   | 0.172 (12.5x)   |
|      cdist, cosine | 1.571   | 1.788   | 1.690   | 0.534 (2.9x)    | 0.618 (2.9x)    | 0.582 (2.9x)    |
|   pdist, euclidean | 1.010   | 1.032   | 1.019   | 0.446 (2.3x)    | 0.524 (2.0x)    | 0.469 (2.2x)    |
| pdist, sqeuclidean | 1.017   | 1.112   | 1.061   | 0.476 (2.1x)    | 0.535 (2.1x)    | 0.505 (2.1x)    |
|      pdist, cosine | 0.751   | 0.845   | 0.789   | 0.560 (1.3x)    | 0.696 (1.2x)    | 0.634 (1.2x)    |
|          euclidean | 0.336   | 0.357   | 0.346   | 0.042 (8.0x)    | 0.047 (7.6x)    | 0.045 (7.7x)    |
|        sqeuclidean | 0.260   | 0.289   | 0.274   | 0.034 (7.6x)    | 0.037 (7.8x)    | 0.035 (7.8x)    |
|             cosine | 0.567   | 0.576   | 0.572   | 0.061 (9.2x)    | 0.067 (8.5x)    | 0.064 (9.0x)    |

#### 64-bit floats

|          Benchmark | Min     | Max     | Mean    | Min (+)         | Max (+)         | Mean (+)        |
|--------------------|---------|---------|---------|-----------------|-----------------|-----------------|
|   cdist, euclidean | 2.003   | 2.159   | 2.065   | 0.316 (6.3x)    | 0.417 (5.2x)    | 0.351 (5.9x)    |
| cdist, sqeuclidean | 2.023   | 2.190   | 2.124   | 0.285 (7.1x)    | 0.411 (5.3x)    | 0.365 (5.8x)    |
|      cdist, cosine | 1.513   | 1.652   | 1.556   | 0.441 (3.4x)    | 0.582 (2.8x)    | 0.482 (3.2x)    |
|   pdist, euclidean | 0.998   | 1.115   | 1.054   | 0.492 (2.0x)    | 0.561 (2.0x)    | 0.523 (2.0x)    |
| pdist, sqeuclidean | 1.013   | 1.169   | 1.060   | 0.532 (1.9x)    | 0.661 (1.8x)    | 0.578 (1.8x)    |
|      pdist, cosine | 0.780   | 0.867   | 0.816   | 0.521 (1.5x)    | 0.647 (1.3x)    | 0.568 (1.4x)    |
|          euclidean | 0.379   | 0.424   | 0.403   | 0.050 (7.6x)    | 0.053 (7.9x)    | 0.052 (7.8x)    |
|        sqeuclidean | 0.275   | 0.295   | 0.286   | 0.049 (5.6x)    | 0.052 (5.7x)    | 0.050 (5.8x)    |
|             cosine | 0.559   | 0.571   | 0.565   | 0.059 (9.5x)    | 0.062 (9.2x)    | 0.060 (9.4x)    |


<table>
<tr>
<th> f32 </th>
<th> f64 </th>
</tr>
<tr>
<td>

![Euclidean f32](images/SIMD-Euclidean_f32.png)
![Squared Euclidean f32](images/SIMD-Squared-Euclidean_f32.png)
![Cosine f32](images/SIMD-Cosine_f32.png)

</td>
<td>

![Euclidean f64](images/SIMD-Euclidean_f64.png)
![Squared Euclidean f64](images/SIMD-Squared-Euclidean_f64.png)
![Cosine f64](images/SIMD-Cosine_f64.png)

</td>
</tr>
</table>

### Vector Distance Benchmarks (No SIMD)

These benchmarks were run on an Intel Core i7-11700KF CPU @ 4.900GHz, using **a single thread**.
The OS was Arch Linux, with kernel version 6.7.4-arch1-1.

The "Min", "Max", and "Mean" columns show the minimum, maximum, and mean times (in seconds), respectively, taken to compute the pairwise distances using the functions from `scipy.spatial.distance`.
The "Min (+)", "Max (+)", and "Mean (+)" columns show the speedup of the this package's functions over the `scipy` functions.
All pairwise distances (`cdist` and `pdist`) were computed for 200x200 vectors of 500 dimensions, and the average time was taken over 100 runs.
All individual distances were computed for 20x20 vectors of 500 dimensions, and the average time was taken over 100 runs.

These benchmarks were run using the `richbench` package.

#### 32-bit Numbers

|          Benchmark | Min     | Max     | Mean    | Min (+)         | Max (+)         | Mean (+)        |
|--------------------|---------|---------|---------|-----------------|-----------------|-----------------|
|         braycurtis | 0.634   | 0.651   | 0.641   | 0.361 (1.8x)    | 0.368 (1.8x)    | 0.365 (1.8x)    |
|           canberra | 1.122   | 1.173   | 1.144   | 0.125 (9.0x)    | 0.128 (9.1x)    | 0.127 (9.0x)    |
|          chebyshev | 1.733   | 1.843   | 1.762   | 0.105 (16.5x)   | 0.117 (15.7x)   | 0.110 (16.1x)   |
|          euclidean | 0.338   | 0.360   | 0.349   | 0.068 (5.0x)    | 0.075 (4.8x)    | 0.070 (5.0x)    |
|        sqeuclidean | 0.259   | 0.263   | 0.261   | 0.067 (3.9x)    | 0.069 (3.8x)    | 0.068 (3.8x)    |
|          cityblock | 0.269   | 0.326   | 0.296   | 0.066 (4.1x)    | 0.074 (4.4x)    | 0.068 (4.3x)    |
|             cosine | 0.567   | 0.608   | 0.576   | 0.242 (2.3x)    | 0.253 (2.4x)    | 0.246 (2.3x)    |
|  cdist, braycurtis | 4.040   | 5.031   | 4.473   | 3.009 (1.3x)    | 4.651 (1.1x)    | 4.232 (1.1x)    |
|    cdist, canberra | 4.181   | 4.290   | 4.231   | 4.571 (-1.1x)   | 4.864 (-1.1x)   | 4.760 (-1.1x)   |
|   cdist, chebyshev | 2.634   | 2.726   | 2.680   | 1.760 (1.5x)    | 1.868 (1.5x)    | 1.815 (1.5x)    |
|   cdist, euclidean | 2.264   | 2.295   | 2.273   | 0.963 (2.4x)    | 1.097 (2.1x)    | 1.002 (2.3x)    |
| cdist, sqeuclidean | 2.228   | 2.292   | 2.256   | 1.000 (2.2x)    | 1.068 (2.1x)    | 1.046 (2.2x)    |
|   cdist, cityblock | 2.243   | 2.271   | 2.256   | 0.731 (3.1x)    | 0.803 (2.8x)    | 0.762 (3.0x)    |
|      cdist, cosine | 1.555   | 1.587   | 1.574   | 3.564 (-2.3x)   | 3.646 (-2.3x)   | 3.610 (-2.3x)   |
|  pdist, braycurtis | 2.048   | 2.327   | 2.151   | 2.609 (-1.3x)   | 3.074 (-1.3x)   | 2.858 (-1.3x)   |
|    pdist, canberra | 1.938   | 2.412   | 2.115   | 2.695 (-1.4x)   | 3.539 (-1.5x)   | 3.096 (-1.5x)   |
|   pdist, chebyshev | 1.249   | 1.259   | 1.253   | 0.935 (1.3x)    | 1.036 (1.2x)    | 0.972 (1.3x)    |
|   pdist, euclidean | 1.006   | 1.123   | 1.032   | 0.687 (1.5x)    | 0.714 (1.6x)    | 0.700 (1.5x)    |
| pdist, sqeuclidean | 1.005   | 1.007   | 1.006   | 0.666 (1.5x)    | 0.894 (1.1x)    | 0.737 (1.4x)    |
|   pdist, cityblock | 0.994   | 1.009   | 1.000   | 0.705 (1.4x)    | 0.827 (1.2x)    | 0.748 (1.3x)    |
|      pdist, cosine | 0.744   | 0.847   | 0.807   | 1.966 (-2.6x)   | 2.452 (-2.9x)   | 2.261 (-2.8x)   |

#### 32-bit Numbers

|          Benchmark | Min     | Max     | Mean    | Min (+)         | Max (+)         | Mean (+)        |
|--------------------|---------|---------|---------|-----------------|-----------------|-----------------|
|         braycurtis | 0.494   | 0.552   | 0.515   | 0.095 (5.2x)    | 0.103 (5.4x)    | 0.098 (5.3x)    |
|           canberra | 0.986   | 1.000   | 0.991   | 0.142 (6.9x)    | 0.147 (6.8x)    | 0.144 (6.9x)    |
|          chebyshev | 1.828   | 2.019   | 1.881   | 0.098 (18.6x)   | 0.102 (19.9x)   | 0.100 (18.9x)   |
|          euclidean | 0.353   | 0.396   | 0.365   | 0.076 (4.7x)    | 0.081 (4.9x)    | 0.077 (4.8x)    |
|        sqeuclidean | 0.265   | 0.299   | 0.276   | 0.077 (3.5x)    | 0.088 (3.4x)    | 0.080 (3.5x)    |
|          cityblock | 0.263   | 0.325   | 0.297   | 0.074 (3.6x)    | 0.077 (4.2x)    | 0.075 (3.9x)    |
|             cosine | 0.594   | 0.730   | 0.667   | 0.256 (2.3x)    | 0.299 (2.4x)    | 0.275 (2.4x)    |
|  cdist, braycurtis | 3.793   | 4.378   | 4.174   | 0.897 (4.2x)    | 1.144 (3.8x)    | 0.992 (4.2x)    |
|    cdist, canberra | 3.860   | 4.008   | 3.915   | 1.850 (2.1x)    | 2.484 (1.6x)    | 2.234 (1.8x)    |
|   cdist, chebyshev | 2.431   | 2.475   | 2.453   | 1.205 (2.0x)    | 1.352 (1.8x)    | 1.300 (1.9x)    |
|   cdist, euclidean | 1.972   | 2.006   | 1.991   | 0.623 (3.2x)    | 0.702 (2.9x)    | 0.666 (3.0x)    |
| cdist, sqeuclidean | 1.952   | 1.967   | 1.961   | 0.642 (3.0x)    | 0.786 (2.5x)    | 0.734 (2.7x)    |
|   cdist, cityblock | 2.004   | 2.094   | 2.028   | 0.658 (3.0x)    | 0.695 (3.0x)    | 0.674 (3.0x)    |
|      cdist, cosine | 1.523   | 1.530   | 1.526   | 2.560 (-1.7x)   | 3.708 (-2.4x)   | 2.961 (-1.9x)   |
|  pdist, braycurtis | 1.985   | 1.994   | 1.989   | 0.760 (2.6x)    | 0.870 (2.3x)    | 0.806 (2.5x)    |
|    pdist, canberra | 1.924   | 1.956   | 1.935   | 1.145 (1.7x)    | 1.216 (1.6x)    | 1.183 (1.6x)    |
|   pdist, chebyshev | 1.266   | 1.306   | 1.279   | 0.930 (1.4x)    | 1.018 (1.3x)    | 0.974 (1.3x)    |
|   pdist, euclidean | 0.995   | 1.043   | 1.008   | 0.655 (1.5x)    | 0.738 (1.4x)    | 0.696 (1.4x)    |
| pdist, sqeuclidean | 0.971   | 1.000   | 0.985   | 0.669 (1.5x)    | 0.741 (1.3x)    | 0.690 (1.4x)    |
|   pdist, cityblock | 1.000   | 1.011   | 1.006   | 0.623 (1.6x)    | 0.718 (1.4x)    | 0.654 (1.5x)    |
|      pdist, cosine | 0.762   | 0.771   | 0.765   | 1.898 (-2.5x)   | 1.969 (-2.6x)   | 1.938 (-2.5x)   |


<table>
<tr>
<th> F32 </th>
<th> F64 </th>
</tr>
<tr>
<td>

![Chebyshev f32](images/Chebyshev_f32.png)
![Euclidean f32](images/Euclidean_f32.png)
![Squared Euclidean f32](images/Squared-Euclidean_f32.png)
![Manhattan f32](images/Manhattan_f32.png)
![Canberra f32](images/Canberra_f32.png)
![Cosine f32](images/Cosine_f32.png)

</td>
<td>

![Chebyshev f64](images/Chebyshev_f64.png)
![Euclidean f64](images/Euclidean_f64.png)
![Squared Euclidean f64](images/Squared-Euclidean_f64.png)
![Manhattan f64](images/Manhattan_f64.png)
![Canberra f64](images/Canberra_f64.png)
![Cosine f64](images/Cosine_f64.png)

</td>
</tr>
</table>

<table>
<tr>
<th> u32 </th>
<th> u64 </th>
</tr>
<tr>
<td>

![Bray-Curtis u32](images/Bray-Curtis_u32.png)

</td>
<td>

![Bray-Curtis u64](images/Bray-Curtis_u64.png)

</td>
</tr>
</table>
</table>

### String Distance Benchmarks

These benchmarks were run on an Intel Core i7-11700KF CPU @ 4.900GHz, using **a single thread**.
The OS was Arch Linux, with kernel version 6.7.4-arch1-1.

All string distances were computed 100 times each, among different pairs of strings, and the average time was taken.

<table>
<tr>
<td>

![Hamming](images/Hamming_str.png)
![Levenshtein](images/Levenshtein_str.png)
![Needleman-Wunsch](images/Needleman-Wunsch_str.png)

</td>
</tr>
</table>

## License

This package is licensed under the MIT license.
