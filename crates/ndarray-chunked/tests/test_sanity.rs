//! Test the rust code.

use ndarray_chunked;

#[test]
fn test_sum_as_string() {
    assert_eq!(ndarray_chunked::sum_as_string(2, 2), "4");
}
