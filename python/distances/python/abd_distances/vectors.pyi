import numpy as np
import numpy.typing as npt

def chebyshev_f32(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> np.float32:
    """Compute the Chebyshev distance between two 1d-arrays."""
    ...

def chebyshev_f64(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> np.float64:
    """Compute the Chebyshev distance between two 1d-arrays."""
    ...

def euclidean_f32(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> np.float32:
    """Compute the Euclidean distance between two 1d-arrays."""
    ...

def euclidean_f64(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> np.float64:
    """Compute the Euclidean distance between two 1d-arrays."""
    ...

def sqeuclidean_f32(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> np.float32:
    """Compute the squared Euclidean distance between two 1d-arrays."""
    ...

def sqeuclidean_f64(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> np.float64:
    """Compute the squared Euclidean distance between two 1d-arrays."""
    ...

def l3_distance_f32(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> np.float32:
    """Compute the L3 distance between two 1d-arrays."""
    ...

def l3_distance_f64(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> np.float64:
    """Compute the L3 distance between two 1d-arrays."""
    ...

def l4_distance_f32(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> np.float32:
    """Compute the L4 distance between two 1d-arrays."""
    ...

def l4_distance_f64(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> np.float64:
    """Compute the L4 distance between two 1d-arrays."""
    ...

def manhattan_f32(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> np.float32:
    """Compute the Manhattan distance between two 1d-arrays."""
    ...

def manhattan_f64(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> np.float64:
    """Compute the Manhattan distance between two 1d-arrays."""
    ...

def braycurtis_u32(a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]) -> np.float32:
    """Compute the Bray-Curtis distance between two 1d-arrays of unsigned integers."""
    ...

def braycurtis_u64(a: npt.NDArray[np.uint64], b: npt.NDArray[np.uint64]) -> np.float64:
    """Compute the Bray-Curtis distance between two 1d-arrays of unsigned integers."""
    ...

def canberra_f32(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> np.float32:
    """Compute the Canberra distance between two 1d-arrays."""
    ...

def canberra_f64(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> np.float64:
    """Compute the Canberra distance between two 1d-arrays."""
    ...

def cosine_f32(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> np.float32:
    """Compute the cosine similarity between two 1d-arrays."""
    ...

def cosine_f64(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> np.float64:
    """Compute the cosine similarity between two 1d-arrays."""
    ...

def hamming_i32(a: npt.NDArray[np.int32], b: npt.NDArray[np.int32]) -> np.float32:
    """Compute the Hamming distance between two 1d-arrays of signed integers."""
    ...

def hamming_i64(a: npt.NDArray[np.int64], b: npt.NDArray[np.int64]) -> np.float64:
    """Compute the Hamming distance between two 1d-arrays of signed integers."""
    ...

def cdist_f32(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    metric: str,
) -> npt.NDArray[np.float32]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def cdist_f64(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    metric: str,
) -> npt.NDArray[np.float64]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def cdist_u32(
    a: npt.NDArray[np.uint32],
    b: npt.NDArray[np.uint32],
    metric: str,
) -> npt.NDArray[np.float32]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def cdist_u64(
    a: npt.NDArray[np.uint64],
    b: npt.NDArray[np.uint64],
    metric: str,
) -> npt.NDArray[np.float64]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def cdist_i32(
    a: npt.NDArray[np.int32],
    b: npt.NDArray[np.int64],
    metric: str,
) -> npt.NDArray[np.uint32]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def cdist_i64(
    a: npt.NDArray[np.int64],
    b: npt.NDArray[np.int64],
    metric: str,
) -> npt.NDArray[np.uint64]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def pdist_f32(
    a: npt.NDArray[np.float32],
    metric: str,
) -> npt.NDArray[np.float32]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def pdist_f64(
    a: npt.NDArray[np.float64],
    metric: str,
) -> npt.NDArray[np.float64]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def pdist_u32(
    a: npt.NDArray[np.uint32],
    metric: str,
) -> npt.NDArray[np.float32]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def pdist_u64(
    a: npt.NDArray[np.uint64],
    metric: str,
) -> npt.NDArray[np.float64]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def pdist_i32(
    a: npt.NDArray[np.int32],
    metric: str,
) -> npt.NDArray[np.uint32]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...

def pdist_i64(
    a: npt.NDArray[np.int64],
    metric: str,
) -> npt.NDArray[np.uint64]:
    """Compute the distance between each pair of the two collections of inputs."""
    ...
