"""Tests for the string distance functions."""


from abd_distances import strings as abd_strings


def test_hamming():
    """Test the Hamming distance function."""
    assert abd_strings.hamming("NAJIBEATSPEPPERS", "NAJIBPEPPERSEATS") == 10
    assert abd_strings.hamming("TOMEATSWHATFOODEATS", "FOODEATSWHATTOMEATS") == 13


def test_levenshtein():
    """Test the Levenshtein distance function."""
    assert abd_strings.levenshtein("NAJIBEATSPEPPERS", "NAJIBPEPPERSEATS") == 8
    assert abd_strings.levenshtein("TOMEATSWHATFOODEATS", "FOODEATSWHATTOMEATS") == 6


def test_needleman_wunsch():
    """Test the Needleman-Wunsch distance function."""
    assert abd_strings.needleman_wunsch("NAJIBEATSPEPPERS", "NAJIBPEPPERSEATS") == 8
    assert abd_strings.needleman_wunsch("TOMEATSWHATFOODEATS", "FOODEATSWHATTOMEATS") == 6  # noqa: E501
