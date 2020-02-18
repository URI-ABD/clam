from typing import Union, Tuple, List

import numpy as np

__all__ = [
    "random",
    "bullseye",
    "line",
    "xor",
    "spiral_2d",
    "tori",
    "skewer",
]

np.random.seed(42)

Data = Union[np.ndarray, np.memmap]
Label = Union[np.ndarray, List[int]]


def random(n: int = 100, dimensions: int = 10) -> Tuple[Data, Label]:
    return np.random.randn(n, dimensions), np.zeros((n, dimensions), dtype=np.int)


def ring_data(n: int, radius: float, noise: float) -> np.ndarray:
    theta: np.ndarray = 2 * np.pi * np.random.rand(n)
    x: np.ndarray = radius * np.cos(theta) + noise * np.random.randn(n)
    y: np.ndarray = radius * np.sin(theta) + noise * np.random.randn(n)
    ring = np.stack([x, y], axis=1)
    return np.asarray(ring, dtype=np.float64)


def bullseye(n: int = 2_000, num_rings: int = 3, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    data: np.ndarray = np.ndarray(shape=(0, 2))
    labels: List[int] = list()
    for r in range(1, 2 * num_rings, 2):
        ring: np.ndarray = ring_data(n=n * r, radius=r, noise=noise)
        labels.extend([r for _ in range(n * r)])
        data = np.concatenate([data, ring], axis=0)
    return np.asarray(data, dtype=np.float64), labels


def line(n: int = 5_000, m: float = 1, c: float = 0., noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    x = np.random.rand(n)
    y = m * x + c
    data = np.asarray((x, y)).T
    data = data + np.random.rand(*data.shape) * noise
    labels = np.ones_like(x.T)
    return np.asarray(data, dtype=np.float64), list(labels)


def xor(n: int = 5_000) -> Tuple[np.ndarray, List[int]]:
    data = np.random.rand(n, 2)
    labels = [int((x > 0.5) != (y > 0.5)) for x, y, in data]
    return np.asarray(data, dtype=np.float64), labels


def spiral_2d(n: int = 5_000, noise: float = 0.1) -> Tuple[np.ndarray, List[int]]:
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n, 2) * noise

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n, 2) * noise

    data = np.concatenate([x_a, x_b]) / 5
    labels = list(np.concatenate([np.zeros(len(x_a)), np.ones(len(x_a))]))
    return np.asarray(data, dtype=np.float64), labels


def generate_torus(n: int, r_torus: float, noise: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_tube: float = r_torus / 5
    u, v = np.random.rand(n), np.random.rand(n)
    u, v = u * 2 * np.pi, v * 2 * np.pi
    x = (r_torus + r_tube * np.cos(v)) * np.cos(u) + (np.random.randn(n) * noise)
    y = (r_torus + r_tube * np.cos(v)) * np.sin(u) + (np.random.randn(n) * noise)
    z = r_tube * np.sin(v) + (np.random.randn(n) * noise)
    return x, y, z


def tori(n: int = 10_000, noise: float = 0.015, r_torus: float = 1.) -> Tuple[np.ndarray, List[int]]:
    x, y, z = generate_torus(n=n // 2, r_torus=r_torus, noise=noise)
    torus_1 = np.stack([x - r_torus, y, z], axis=1)
    labels = [0 for _ in x]

    x, y, z = generate_torus(n=n // 2, r_torus=r_torus, noise=noise)
    torus_2 = np.stack([x, z, y], axis=1)
    labels.extend([1 for _ in x])

    data = np.concatenate([torus_1, torus_2], axis=0)
    return np.asarray(data, dtype=np.float64), labels


def spiral_3d(n: int, radius: float, height: float, num_turns: int, noise: float) -> np.ndarray:
    theta: np.ndarray = 2 * np.pi * np.random.rand(n) * num_turns
    x: np.ndarray = radius * np.cos(theta) + noise * np.random.randn(n)
    y: np.ndarray = radius * np.sin(theta) + noise * np.random.randn(n)
    z: np.ndarray = height * theta / (2 * np.pi * num_turns) + noise * np.random.randn(n)
    return np.stack([x, y, z], axis=1)


def line_3d(n: int, height: float, noise: float) -> np.ndarray:
    x: np.ndarray = noise * np.random.randn(n)
    y: np.ndarray = noise * np.random.randn(n)
    z: np.ndarray = height * np.random.rand(n)
    return np.stack([x, y, z], axis=1)


def skewer(
        n: int = 10_000,
        radius: float = 3.,
        height: float = 6.,
        num_turns: int = 2,
        noise: float = 0.05
) -> Tuple[np.ndarray, List[int]]:
    spiral_points, line_points = 5 * n // 6, n // 6
    spiral_data = spiral_3d(spiral_points, radius, height, num_turns, noise)
    labels = [0 for _ in range(spiral_data.shape[0])]

    line_data = line_3d(line_points, height, noise)
    labels.extend([1 for _ in range(line_data.shape[0])])

    data = np.concatenate([spiral_data, line_data], axis=0)
    return np.asarray(data, dtype=np.float64), labels
