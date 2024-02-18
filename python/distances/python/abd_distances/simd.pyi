import numpy as np
import numpy.typing as npt


def euclidean_f32(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> np.float32:
    """Compute the Euclidean distance between two 1d-arrays."""
    ...

def euclidean_f64(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> np.float64:
    """Compute the Euclidean distance between two 1d-arrays."""
    ...

def euclidean_sq_f32(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> np.float32:
    """Compute the squared Euclidean distance between two 1d-arrays."""
    ...

def euclidean_sq_f64(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> np.float64:
    """Compute the squared Euclidean distance between two 1d-arrays."""
    ...

def cosine_f32(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> np.float32:
    """Compute the cosine similarity between two 1d-arrays."""
    ...

def cosine_f64(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> np.float64:
    """Compute the cosine similarity between two 1d-arrays."""
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
