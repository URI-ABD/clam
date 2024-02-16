# Algorithms for Big Data: Distances (v0.1.1-dev0)

This package contains algorithms for computing distances between data points.
It is a thin Python wrapper around the `distances` crate, in Rust.

## Installation

```bash
pip install abd_distances@0.1.1-dev0
```

## Usage

```python
import numpy

from abd_distances.simd import euclidean_f32

a = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
b = a + 3.0

distance = euclidean_f32(a, b)

print(distance)
```

## Benchmarks

These benchmarks were run on an Intel Core i7-11700KF CPU @ 4.900GHz, using a single thread.
The OS was Arch Linux, with kernel version 6.7.4-arch1-1.

### SIMD-Accelerated Vector Distances

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
![L3 f32](images/L3_f32.png)
![L4 f32](images/L4_f32.png)
![Manhattan f32](images/Manhattan_f32.png)
![Canberra f32](images/Canberra_f32.png)
![Cosine f32](images/Cosine_f32.png)

</td>
<td>

![Chebyshev f64](images/Chebyshev_f64.png)
![Euclidean f64](images/Euclidean_f64.png)
![Squared Euclidean f64](images/Squared-Euclidean_f64.png)
![L3 f64](images/L3_f64.png)
![L4 f64](images/L4_f64.png)
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

<table>
<tr>
<th> i32 </th>
<th> i64 </th>
</tr>
<tr>
<td>

![Hamming i32](images/Hamming_i32.png)

</td>
<td>

![Hamming i64](images/Hamming_i64.png)

</td>
</tr>
</table>

### String Distance Benchmarks

These benchmarks were run on an Intel Core i7-11700KF CPU @ 4.900GHz, using a single thread.
The OS was Arch Linux, with kernel version 6.7.4-arch1-1.

All string distances were computed 100 times each, among different pairs of strings, and the average time was taken.

![Hamming](images/Hamming_str.png)
![Levenshtein](images/Levenshtein_str.png)
![Needleman-Wunsch](images/Needleman-Wunsch_str.png)


## License

This package is licensed under the MIT license.
