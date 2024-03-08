"""Tests for the string distance functions."""

import editdistance
from abd_distances import strings as abd_strings


def test_hamming(strings: list[str]):
    """Test the Hamming distance function."""
    assert abd_strings.hamming("NAJIBEATSPEPPERS", "NAJIBPEPPERSEATS") == 10
    assert abd_strings.hamming("TOMEATSWHATFOODEATS", "FOODEATSWHATTOMEATS") == 13

    for a in strings:
        for b in strings:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                dist = abd_strings.hamming(a_, b_)
                expected = sum(1 for i in range(len(a_)) if a_[i] != b_[i])
                assert dist == expected, f"Expected: {expected}, got: {dist}"


def test_levenshtein(strings: list[str]):
    """Test the Levenshtein distance function."""
    assert abd_strings.levenshtein("NAJIBEATSPEPPERS", "NAJIBPEPPERSEATS") == 8
    assert abd_strings.levenshtein("TOMEATSWHATFOODEATS", "FOODEATSWHATTOMEATS") == 6

    for a in strings:
        for b in strings:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                dist = abd_strings.levenshtein(a_, b_)
                expected = editdistance.eval(a_, b_)
                assert dist == expected, f"Expected: {expected}, got: {dist}"


def test_needleman_wunsch(strings: list[str]):
    """Test the Needleman-Wunsch distance function."""
    assert abd_strings.needleman_wunsch("NAJIBEATSPEPPERS", "NAJIBPEPPERSEATS") == 8
    assert abd_strings.needleman_wunsch("TOMEATSWHATFOODEATS", "FOODEATSWHATTOMEATS") == 6

    for a in strings:
        for b in strings:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                dist = abd_strings.needleman_wunsch(a_, b_)
                expected = editdistance.eval(a_, b_)
                assert dist == expected, f"Expected: {expected}, got: {dist}"
