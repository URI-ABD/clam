# Algorithms for Big Data: Distances

This package contains algorithms for computing distances between data points.
It is a thin Python wrapper around the `distances` crate, in Rust.

## Installation

```bash
pip install distances@0.1.1-dev0
```

## Usage

```python
from distances.vectors import euclidean_f32

a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

distance = euclidean_f32(a, b)

print(distance)
```

## License

This package is licensed under the MIT license.
