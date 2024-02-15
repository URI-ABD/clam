# Algorithms for Big Data: Distances

This package contains algorithms for computing distances between data points.
It is a thin Python wrapper around the `distances` crate, in Rust.

## Installation

```bash
pip install distances@0.1.1-dev0
```

## Usage

```python
import numpy

from distances.vectors import euclidean_f32

a = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
b = a + 3.0

distance = euclidean_f32(a, b)

print(distance)
```

## Benchmarks

These benchmarks were run on an Intel Core i7-11700KF CPU @ 4.900GHz, using a single thread.
The OS was Arch Linux, with kernel version 6.7.4-arch1-1.

### SIMD-Accelerated Vector Distances

These benchmarks were run on vectors of 1,000 dimensions, and each benchmark was run at least 10,000 times.
The multiplication factor, in the Min, Max, and Mean columns, is the factor by which the Rust implementation is faster than the SciPy implementation.

|              Benchmark | Min     | Max     | Mean    | Min (+)         | Max (+)         | Mean (+)        |
|------------------------|---------|---------|---------|-----------------|-----------------|-----------------|
|         Euclidean, f32 | 0.201   | 0.216   | 0.205   | 0.018 (10.9x)   | 0.019 (11.6x)   | 0.019 (11.1x)   |
|         Euclidean, f64 | 0.217   | 0.241   | 0.226   | 0.024 (9.1x)    | 0.024 (10.1x)   | 0.024 (9.5x)    |
| Euclidean squared, f32 | 0.151   | 0.155   | 0.152   | 0.019 (8.1x)    | 0.019 (8.2x)    | 0.019 (8.1x)    |
| Euclidean squared, f64 | 0.155   | 0.160   | 0.157   | 0.024 (6.5x)    | 0.024 (6.7x)    | 0.024 (6.6x)    |
|            Cosine, f32 | 0.189   | 0.190   | 0.189   | 0.033 (5.6x)    | 0.034 (5.7x)    | 0.033 (5.7x)    |
|            Cosine, f64 | 0.140   | 0.141   | 0.140   | 0.035 (3.9x)    | 0.035 (4.0x)    | 0.035 (4.0x)    |

### Vector Distance Benchmarks (No SIMD)

These benchmarks were run on vectors of 1,000 dimensions, and each benchmark was run at least 10,000 times.
The multiplication factor, in the Min, Max, and Mean columns, is the factor by which the Rust implementation is faster than the SciPy implementation.

|              Benchmark | Min     | Max     | Mean    | Min (+)         | Max (+)         | Mean (+)        |
|------------------------|---------|---------|---------|-----------------|-----------------|-----------------|
|         Chebyshev, f32 | 1.649   | 1.709   | 1.668   | 0.056 (29.4x)   | 0.057 (30.2x)   | 0.056 (29.6x)   |
|         Chebyshev, f64 | 1.689   | 1.736   | 1.705   | 0.056 (30.0x)   | 0.056 (30.8x)   | 0.056 (30.3x)   |
|         Euclidean, f32 | 0.201   | 0.202   | 0.201   | 0.056 (3.6x)    | 0.057 (3.6x)    | 0.056 (3.6x)    |
|         Euclidean, f64 | 0.211   | 0.214   | 0.212   | 0.056 (3.7x)    | 0.057 (3.8x)    | 0.057 (3.7x)    |
| Euclidean squared, f32 | 0.151   | 0.153   | 0.152   | 0.056 (2.7x)    | 0.056 (2.7x)    | 0.056 (2.7x)    |
| Euclidean squared, f64 | 0.157   | 0.160   | 0.158   | 0.056 (2.8x)    | 0.057 (2.8x)    | 0.056 (2.8x)    |
|                L3, f32 | 0.452   | 0.456   | 0.453   | 0.057 (7.9x)    | 0.058 (7.9x)    | 0.057 (7.9x)    |
|                L3, f64 | 0.524   | 0.555   | 0.535   | 0.059 (8.8x)    | 0.061 (9.2x)    | 0.060 (8.9x)    |
|                L4, f32 | 0.451   | 0.453   | 0.453   | 0.056 (8.0x)    | 0.057 (8.0x)    | 0.056 (8.0x)    |
|                L4, f64 | 0.519   | 0.523   | 0.522   | 0.057 (9.2x)    | 0.057 (9.2x)    | 0.057 (9.2x)    |
|         Manhattan, f32 | 0.136   | 0.137   | 0.136   | 0.056 (2.4x)    | 0.056 (2.4x)    | 0.056 (2.4x)    |
|         Manhattan, f64 | 0.146   | 0.148   | 0.147   | 0.056 (2.6x)    | 0.056 (2.6x)    | 0.056 (2.6x)    |
|       Bray-Curtis, u32 | 0.365   | 0.366   | 0.365   | 0.022 (16.5x)   | 0.022 (16.5x)   | 0.022 (16.5x)   |
|       Bray-Curtis, u64 | 0.401   | 0.403   | 0.402   | 0.042 (9.6x)    | 0.043 (9.4x)    | 0.042 (9.5x)    |
|          Canberra, f32 | 0.834   | 0.838   | 0.836   | 0.057 (14.5x)   | 0.057 (14.6x)   | 0.057 (14.6x)   |
|          Canberra, f64 | 0.741   | 0.745   | 0.743   | 0.056 (13.2x)   | 0.056 (13.2x)   | 0.056 (13.2x)   |
|            Cosine, f32 | 0.189   | 0.191   | 0.190   | 0.139 (1.4x)    | 0.140 (1.4x)    | 0.139 (1.4x)    |
|            Cosine, f64 | 0.137   | 0.137   | 0.137   | 0.133 (1.0x)    | 0.134 (1.0x)    | 0.133 (1.0x)    |
|           Hamming, i32 | 0.282   | 0.285   | 0.284   | 0.017 (16.6x)   | 0.017 (16.7x)   | 0.017 (16.7x)   |
|           Hamming, i64 | 0.291   | 0.294   | 0.292   | 0.023 (12.6x)   | 0.023 (12.7x)   | 0.023 (12.7x)   |

## License

This package is licensed under the MIT license.
