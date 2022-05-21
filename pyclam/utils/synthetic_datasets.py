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
    numpy.random.seed(42)

    data = numpy.random.randn(n, dimensions)
    labels = [0 for _ in range(n)]
    return data, labels


def ring_data(n: int, radius: float, noise: float) -> numpy.ndarray:
    numpy.random.seed(42)

    theta: numpy.ndarray = 2 * numpy.pi * numpy.random.rand(n)
    x: numpy.ndarray = radius * numpy.cos(theta) + noise * numpy.random.randn(n)
    y: numpy.ndarray = radius * numpy.sin(theta) + noise * numpy.random.randn(n)
    ring = numpy.stack([x, y], axis=1)
    return numpy.asarray(ring, dtype=numpy.float64)


def bullseye(n: int = 2_000, num_rings: int = 3, noise: float = 0.05) -> Data:
    numpy.random.seed(42)

    data = numpy.ndarray(shape=(0, 2))
    labels = list()
    for r in range(1, 2 * num_rings, 2):
        ring: numpy.ndarray = ring_data(n=n * r, radius=r, noise=noise)
        labels.extend([r for _ in range(n * r)])
        data = numpy.concatenate([data, ring], axis=0)
    return numpy.asarray(data, dtype=numpy.float64), labels


def line(n: int = 5_000, m: float = 1, c: float = 0., noise: float = 0.05) -> Data:
    numpy.random.seed(42)

    x = numpy.random.rand(n)
    y = m * x + c
    data = numpy.asarray((x, y)).T
    data = data + numpy.random.rand(*data.shape) * noise
    labels = numpy.ones_like(x.T)
    return numpy.asarray(data, dtype=numpy.float64), list(labels)


def xor(n: int = 5_000) -> Data:
    numpy.random.seed(42)

    data = numpy.random.rand(n, 2)
    labels = [int((x > 0.5) != (y > 0.5)) for x, y, in data]
    return numpy.asarray(data, dtype=numpy.float64), labels


def spiral_2d(n: int = 5_000, noise: float = 0.1) -> Data:
    numpy.random.seed(42)

    theta = numpy.sqrt(numpy.random.rand(n)) * 2 * numpy.pi

    r_a = 2 * theta + numpy.pi
    data_a = numpy.array([numpy.cos(theta) * r_a, numpy.sin(theta) * r_a]).T
    x_a = data_a + numpy.random.randn(n, 2) * noise

    r_b = -2 * theta - numpy.pi
    data_b = numpy.array([numpy.cos(theta) * r_b, numpy.sin(theta) * r_b]).T
    x_b = data_b + numpy.random.randn(n, 2) * noise

    data = numpy.concatenate([x_a, x_b]) / 5
    labels = list(numpy.concatenate([numpy.zeros(len(x_a)), numpy.ones(len(x_a))]))
    return numpy.asarray(data, dtype=numpy.float64), labels


def generate_torus(n: int, r_torus: float, noise: float) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    numpy.random.seed(42)

    r_tube: float = r_torus / 5
    u, v = numpy.random.rand(n), numpy.random.rand(n)
    u, v = u * 2 * numpy.pi, v * 2 * numpy.pi
    x = (r_torus + r_tube * numpy.cos(v)) * numpy.cos(u) + (numpy.random.randn(n) * noise)
    y = (r_torus + r_tube * numpy.cos(v)) * numpy.sin(u) + (numpy.random.randn(n) * noise)
    z = r_tube * numpy.sin(v) + (numpy.random.randn(n) * noise)
    return x, y, z


def tori(n: int = 10_000, noise: float = 0.015, r_torus: float = 1.) -> Data:
    numpy.random.seed(42)

    x, y, z = generate_torus(n=n // 2, r_torus=r_torus, noise=noise)
    torus_1 = numpy.stack([x - r_torus, y, z], axis=1)
    labels = [0 for _ in x]

    x, y, z = generate_torus(n=n // 2, r_torus=r_torus, noise=noise)
    torus_2 = numpy.stack([x, z, y], axis=1)
    labels.extend([1 for _ in x])

    data = numpy.concatenate([torus_1, torus_2], axis=0)
    return numpy.asarray(data, dtype=numpy.float64), labels


def spiral_3d(n: int, radius: float, height: float, num_turns: int, noise: float) -> numpy.ndarray:
    numpy.random.seed(42)

    theta: numpy.ndarray = 2 * numpy.pi * numpy.random.rand(n) * num_turns
    x: numpy.ndarray = radius * numpy.cos(theta) + noise * numpy.random.randn(n)
    y: numpy.ndarray = radius * numpy.sin(theta) + noise * numpy.random.randn(n)
    z: numpy.ndarray = height * theta / (2 * numpy.pi * num_turns) + noise * numpy.random.randn(n)
    return numpy.stack([x, y, z], axis=1)


def line_3d(n: int, height: float, noise: float) -> numpy.ndarray:
    numpy.random.seed(42)

    x: numpy.ndarray = noise * numpy.random.randn(n)
    y: numpy.ndarray = noise * numpy.random.randn(n)
    z: numpy.ndarray = height * numpy.random.rand(n)
    return numpy.stack([x, y, z], axis=1)


def skewer(
        n: int = 10_000,
        radius: float = 3.,
        height: float = 6.,
        num_turns: int = 2,
        noise: float = 0.05
) -> Data:
    numpy.random.seed(42)

    spiral_points, line_points = 5 * n // 6, n // 6
    spiral_data = spiral_3d(spiral_points, radius, height, num_turns, noise)
    labels = [0 for _ in range(spiral_data.shape[0])]

    line_data = line_3d(line_points, height, noise)
    labels.extend([1 for _ in range(line_data.shape[0])])

    data = numpy.concatenate([spiral_data, line_data], axis=0)
    return numpy.asarray(data, dtype=numpy.float64), labels
