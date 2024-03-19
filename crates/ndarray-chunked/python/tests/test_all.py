# ruff: noqa: S311

import random
import tempfile
from pathlib import Path

import numpy as np
from ndarray_chunked import ChunkedArrayI64  # type: ignore[attr-defined]
from ndarray_chunked import chunk_i64  # type: ignore[attr-defined]
from numpy.testing import assert_array_equal


def generate_dummy_data(
    dirname: str,
    chunk_along: int,
    size: int,
    shape: tuple[int, int, int],
) -> tuple[Path, np.ndarray]:
    # Dummy array
    arr = np.arange(0, np.prod(shape)).reshape(shape)

    # Create our temp directory and settings
    tmp = Path(tempfile.mkdtemp())

    # Write out the chunks
    chunk_i64(arr, chunk_along, size, str(tmp / dirname))

    return tmp, arr


def test_large_array():
    chunk_along = 0
    size = 3
    shape = (100, 100, 100)

    # The number of files is the number of chunks + 1 for the metadata
    filenum = shape[chunk_along] // size + 1
    dirname = "test_chunked_arr_large"

    tmp, _ = generate_dummy_data(dirname, chunk_along, size, shape)

    # Assert that the correct number of chunks were written
    files = list((tmp / dirname).iterdir())

    assert len(files) - 1 == filenum

    ca = ChunkedArrayI64(str(tmp / dirname))
    assert ca.shape() == list(shape)


def test_get_single_chunk():
    chunk_along = 0
    size = 3
    shape = (9, 3, 3)
    dirname = "test_chunked_arr"

    tmp, arr = generate_dummy_data(dirname, chunk_along, size, shape)
    ca = ChunkedArrayI64(str(tmp / dirname))

    slice_ = (slice(0, 3), slice(None), slice(None))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)


def test_get_multiple_chunks():
    chunk_along = 0
    size = 3
    shape = (9, 3, 3)
    dirname = "test_chunked_arr"

    tmp, arr = generate_dummy_data(dirname, chunk_along, size, shape)
    ca = ChunkedArrayI64(str(tmp / dirname))

    slice_ = (slice(0, 9), slice(None), slice(None))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)


def test_get_sliced():
    chunk_along = 0
    size = 3
    shape = (9, 3, 3)
    dirname = "test_chunked_arr"

    tmp, arr = generate_dummy_data(dirname, chunk_along, size, shape)
    ca = ChunkedArrayI64(str(tmp / dirname))

    slice_ = (slice(1, 4), slice(1, 3), slice(1, 3))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(expected, result)


def test_get_random_slice():
    chunk_along = 0
    size = 3
    shape = (9, 3, 3)
    filenum = shape[chunk_along] // size + 1
    dirname = "test_chunked_arr"

    tmp, arr = generate_dummy_data(dirname, chunk_along, size, shape)

    # Assert that the correct number of chunks were written
    files = list((tmp / dirname).iterdir())

    assert len(files) == filenum

    ca = ChunkedArrayI64(str(tmp / dirname))
    assert ca.shape() == list(shape)

    # Smoke test
    assert_array_equal(arr, ca[slice(None), slice(None), slice(None)])

    # A randomized slice of our 3d array
    slice_ = (
        slice(
            random.randint(0, shape[0] // 2),
            random.randint(shape[0] // 2, shape[0]),
        ),
        slice(
            random.randint(0, shape[1] // 2),
            random.randint(shape[1] // 2, shape[1]),
        ),
        slice(
            random.randint(0, shape[2] // 2),
            random.randint(shape[2] // 2, shape[2]),
        ),
    )
    # Assert they're equal
    have = arr[slice_]
    got = ca[slice_]

    assert got.shape == have.shape
    assert_array_equal(got, have)


def test_get_random_slice_stepped():
    chunk_along = 0
    size = 3
    shape = (9, 3, 3)
    filenum = shape[chunk_along] // size + 1
    dirname = "test_chunked_arr"

    tmp, arr = generate_dummy_data(dirname, chunk_along, size, shape)

    # Assert that the correct number of chunks were written
    files = list((tmp / dirname).iterdir())

    assert len(files) == filenum

    ca = ChunkedArrayI64(str(tmp / dirname))
    assert ca.shape() == list(shape)

    # Smoke test
    assert_array_equal(arr, ca[slice(None), slice(None), slice(None)])

    # A randomized slice of our 3d array
    slice_ = (
        slice(
            random.randint(0, shape[0] // 2),
            random.randint(shape[0] // 2, shape[0]),
            random.randint(1, shape[0]),
        ),
        slice(
            random.randint(0, shape[1] // 2),
            random.randint(shape[1] // 2, shape[1]),
            random.randint(1, shape[1]),
        ),
        slice(
            random.randint(0, shape[2] // 2),
            random.randint(shape[2] // 2, shape[2]),
            random.randint(1, shape[2]),
        ),
    )
    # Assert they're equal
    have = arr[slice_]
    got = ca[slice_]

    assert got.shape == have.shape
    assert_array_equal(got, have)


def test_general():
    chunk_along = 0
    size = 3
    shape = (9, 3, 3)
    dirname = "test_chunked_arr"

    tmp, arr = generate_dummy_data(dirname, chunk_along, size, shape)
    ca = ChunkedArrayI64(str(tmp / dirname))

    # Smoke test
    assert_array_equal(arr, ca[slice(None), slice(None), slice(None)])

    # Edge cases
    slice_ = (slice(0, 1), slice(0, 1), slice(0, 1))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)

    slice_ = (slice(0, 1), slice(0, 1), slice(0, 3))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)

    slice_ = (slice(0, 1), slice(0, 3), slice(0, 3))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)

    slice_ = (slice(0, 1), slice(0, 3), slice(0, 1))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)

    slice_ = (slice(0, 1), slice(0, 3), slice(0, 2))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)

    slice_ = (slice(0, 1), slice(0, 2), slice(0, 3))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)

    slice_ = (slice(0, 1), slice(0, 2), slice(0, 1))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)

    slice_ = (slice(0, 1), slice(0, 2), slice(0, 2))
    expected = arr[slice_]
    result = ca[slice_]

    assert_array_equal(result, expected)
