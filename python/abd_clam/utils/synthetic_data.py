"""Synthetic data generation tests and examples in CLAM."""

import numpy

__all__ = [
    "random",
    "bullseye",
    "line",
    "xor",
    "spiral_2d",
    "tori",
    "skewer",
]

Data = tuple[numpy.ndarray, list[int]]


def random(n: int = 100, dimensions: int = 10) -> Data:
    """Generate a 2D random dataset."""
    rng = numpy.random.default_rng()
    data = rng.normal(size=(n, dimensions))
    labels = [0 for _ in range(n)]
    return data, labels


def ring_data(n: int, radius: float, noise: float) -> numpy.ndarray:
    """Generate a 2D ring dataset."""
    rng = numpy.random.default_rng()
    theta = 2 * numpy.pi * rng.random(n)
    x = radius * numpy.cos(theta) + noise * rng.normal(size=n)
    y = radius * numpy.sin(theta) + noise * rng.normal(size=n)
    ring = numpy.stack([x, y], axis=1)
    return numpy.asarray(ring, dtype=numpy.float64)


def bullseye(n: int = 2_000, num_rings: int = 3, noise: float = 0.05) -> Data:
    """Generate a 2D bullseye dataset."""
    data = numpy.ndarray(shape=(0, 2))
    labels = []
    for i, r in enumerate(range(1, 2 * num_rings, 2)):
        ring: numpy.ndarray = ring_data(n=n * r, radius=r, noise=noise)
        labels.extend([i for _ in range(n * r)])
        data = numpy.concatenate([data, ring], axis=0)
    return numpy.asarray(data, dtype=numpy.float64), labels


def line(n: int = 5_000, m: float = 1, c: float = 0.0, noise: float = 0.05) -> Data:
    """Generate a 2D line dataset."""
    rng = numpy.random.default_rng()
    x = rng.random(n)
    y = m * x + c
    data = numpy.asarray((x, y)).T
    data = data + rng.random(data.shape) * noise
    labels = numpy.ones_like(x.T)
    return numpy.asarray(data, dtype=numpy.float64), list(labels)


def xor(n: int = 5_000) -> Data:
    """Generate a 2D XOR dataset."""
    rng = numpy.random.default_rng()
    data = rng.random(n)
    labels = [int((x > 0.5) != (y > 0.5)) for x, y, in data]
    return numpy.asarray(data, dtype=numpy.float64), labels


def spiral_2d(n: int = 5_000, noise: float = 0.1) -> Data:
    """Generate a 2D spiral dataset."""
    rng = numpy.random.default_rng()

    theta = numpy.sqrt(rng.random(n)) * 2 * numpy.pi

    r_a = 2 * theta + numpy.pi
    data_a = numpy.array([numpy.cos(theta) * r_a, numpy.sin(theta) * r_a]).T
    x_a = data_a + rng.normal(size=n) * noise

    r_b = -2 * theta - numpy.pi
    data_b = numpy.array([numpy.cos(theta) * r_b, numpy.sin(theta) * r_b]).T
    x_b = data_b + rng.normal(size=n) * noise

    data = numpy.concatenate([x_a, x_b]) / 5
    labels = list(numpy.concatenate([numpy.zeros(len(x_a)), numpy.ones(len(x_a))]))
    return numpy.asarray(data, dtype=numpy.float64), labels


def generate_torus(
    n: int,
    r_torus: float,
    noise: float,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Generate a 3D torus."""
    rng = numpy.random.default_rng()

    r_tube: float = r_torus / 5
    u, v = rng.random(n), rng.random(n)
    u, v = u * 2 * numpy.pi, v * 2 * numpy.pi
    x = (r_torus + r_tube * numpy.cos(v)) * numpy.cos(u) + (rng.normal(size=n) * noise)
    y = (r_torus + r_tube * numpy.cos(v)) * numpy.sin(u) + (rng.normal(size=n) * noise)
    z = r_tube * numpy.sin(v) + (rng.normal(size=n) * noise)
    return x, y, z


def tori(n: int = 10_000, noise: float = 0.015, r_torus: float = 1.0) -> Data:
    """Generate a 3D tori dataset."""
    x, y, z = generate_torus(n=n // 2, r_torus=r_torus, noise=noise)
    torus_1 = numpy.stack([x - r_torus, y, z], axis=1)
    labels = [0 for _ in x]

    x, y, z = generate_torus(n=n // 2, r_torus=r_torus, noise=noise)
    torus_2 = numpy.stack([x, z, y], axis=1)
    labels.extend([1 for _ in x])

    data = numpy.concatenate([torus_1, torus_2], axis=0)
    return numpy.asarray(data, dtype=numpy.float64), labels


def spiral_3d(
    n: int,
    radius: float,
    height: float,
    num_turns: int,
    noise: float,
) -> numpy.ndarray:
    """Generate a 3D spiral dataset."""
    rng = numpy.random.default_rng()

    theta = 2 * numpy.pi * num_turns * rng.random(n)
    x = radius * numpy.cos(theta) + noise * rng.normal(size=n)
    y = radius * numpy.sin(theta) + noise * rng.normal(size=n)
    z = height * theta / (2 * numpy.pi * num_turns) + noise * rng.normal(size=n)
    return numpy.stack([x, y, z], axis=1)


def line_3d(n: int, height: float, noise: float) -> numpy.ndarray:
    """Generate a 3D line dataset."""
    rng = numpy.random.default_rng()

    x = noise * rng.normal(size=n)
    y = noise * rng.normal(size=n)
    z = height * rng.random(n)
    return numpy.stack([x, y, z], axis=1)


def skewer(
    n: int = 10_000,
    radius: float = 3.0,
    height: float = 6.0,
    num_turns: int = 2,
    noise: float = 0.05,
) -> Data:
    """Generate a 3D skewer dataset."""
    spiral_points, line_points = 5 * n // 6, n // 6
    spiral_data = spiral_3d(spiral_points, radius, height, num_turns, noise)
    labels = [0 for _ in range(spiral_data.shape[0])]

    line_data = line_3d(line_points, height, noise)
    labels.extend([1 for _ in range(line_data.shape[0])])

    data = numpy.concatenate([spiral_data, line_data], axis=0)
    return numpy.asarray(data, dtype=numpy.float64), labels
