import ndarray_chunked


def test_sum_as_string():
    assert ndarray_chunked.sum_as_string(1, 1) == "2"
