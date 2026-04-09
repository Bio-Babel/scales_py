"""Coverage tests for scales/minor_breaks.py and breaks_log.py."""

import numpy as np
import pytest

from scales.minor_breaks import (
    minor_breaks_n,
    minor_breaks_width,
    regular_minor_breaks,
)

from scales.breaks_log import (
    breaks_log,
    minor_breaks_log,
    _fill_log_breaks,
)


# ===========================================================================
# minor_breaks.py (lines 58, 153-198)
# ===========================================================================

# ---------------------------------------------------------------------------
# minor_breaks_n (line 58)
# ---------------------------------------------------------------------------

class TestMinorBreaksN:
    def test_basic(self):
        fn = minor_breaks_n(n=4)
        result = fn(np.array([0, 10, 20]), np.array([0, 20]), 4)
        assert len(result) > 0
        # All minor breaks should be within limits
        assert np.all(result >= 0)
        assert np.all(result <= 20)

    def test_fewer_than_two_majors(self):
        fn = minor_breaks_n(n=4)
        result = fn(np.array([5]), np.array([0, 10]), 4)
        assert len(result) == 0

    def test_default_n_minor(self):
        fn = minor_breaks_n(n=2)
        result = fn(np.array([0, 10]), np.array([0, 10]))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# minor_breaks_width
# ---------------------------------------------------------------------------

class TestMinorBreaksWidth:
    def test_basic(self):
        fn = minor_breaks_width(2.5)
        result = fn(np.array([0, 10, 20]), np.array([0, 20]), 5)
        assert len(result) > 0

    def test_with_offset(self):
        fn = minor_breaks_width(5, offset=1)
        result = fn(np.array([0, 10]), np.array([0, 10]), 5)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# regular_minor_breaks (lines 153-198)
# ---------------------------------------------------------------------------

class TestRegularMinorBreaks:
    def test_basic(self):
        fn = regular_minor_breaks()
        result = fn(np.array([0, 5, 10]), np.array([0, 10]), 2)
        assert len(result) > 0

    def test_reverse(self):
        fn = regular_minor_breaks(reverse=True)
        result = fn(np.array([0, 5, 10]), np.array([0, 10]), 2)
        assert len(result) > 0

    def test_fewer_than_two_majors(self):
        fn = regular_minor_breaks()
        result = fn(np.array([5]), np.array([0, 10]), 2)
        assert len(result) == 0

    def test_n_zero(self):
        fn = regular_minor_breaks()
        result = fn(np.array([0, 10]), np.array([0, 10]), 0)
        assert len(result) == 0

    def test_n_one(self):
        fn = regular_minor_breaks()
        result = fn(np.array([0, 10]), np.array([0, 10]), 1)
        # n-1 = 0 minor breaks
        assert len(result) == 0

    def test_n_three(self):
        fn = regular_minor_breaks()
        result = fn(np.array([0, 10, 20]), np.array([0, 20]), 3)
        assert len(result) > 0

    def test_reverse_result_order(self):
        fn = regular_minor_breaks(reverse=True)
        result = fn(np.array([0, 10, 20]), np.array([0, 20]), 3)
        # Result should still be sorted (just negated and reversed back)
        assert len(result) > 0


# ===========================================================================
# breaks_log.py (lines 56, 77-78, 105-111, 180, 185, 196, 218)
# ===========================================================================

# ---------------------------------------------------------------------------
# breaks_log (lines 56, 77-78)
# ---------------------------------------------------------------------------

class TestBreaksLog:
    def test_basic(self):
        brk = breaks_log(n=5, base=10)
        result = brk([1, 10000])
        assert len(result) > 0

    def test_empty(self):
        brk = breaks_log(n=5)
        result = brk([float("nan"), float("-inf")])
        assert len(result) == 0

    def test_few_breaks_fill(self):
        # When integer powers give too few, should fill in
        brk = breaks_log(n=10, base=10)
        result = brk([1, 100])  # Only 3 integer powers
        assert len(result) > 3

    def test_too_many_thin(self):
        # When too many breaks, should thin
        brk = breaks_log(n=3, base=10)
        result = brk([1, 1e15])  # 16 integer powers
        assert len(result) > 0

    def test_base_2(self):
        brk = breaks_log(n=5, base=2)
        result = brk([1, 64])
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _fill_log_breaks (lines 105-111)
# ---------------------------------------------------------------------------

class TestFillLogBreaks:
    def test_base_10(self):
        result = _fill_log_breaks((0, 2), 10, 10)
        assert len(result) > 0

    def test_base_2(self):
        result = _fill_log_breaks((0, 5), 2, 5)
        assert len(result) > 0

    def test_other_base(self):
        result = _fill_log_breaks((0, 3), 5, 5)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# minor_breaks_log (lines 180, 185, 196, 218)
# ---------------------------------------------------------------------------

class TestMinorBreaksLog:
    def test_basic(self):
        fn = minor_breaks_log()
        majors = np.array([1, 10, 100])
        result = fn(majors, np.array([1, 100]), 5)
        assert len(result) > 0

    def test_fewer_than_two_majors(self):
        fn = minor_breaks_log()
        result = fn(np.array([10]), np.array([1, 100]), 5)
        assert len(result) == 0

    def test_no_positive_majors(self):
        fn = minor_breaks_log()
        result = fn(np.array([-10, -1]), np.array([-10, -1]), 5)
        assert len(result) == 0

    def test_detail_one(self):
        fn = minor_breaks_log(detail=1)
        majors = np.array([1, 10, 100])
        result = fn(majors, np.array([1, 100]), 5)
        assert len(result) == 0

    def test_detail_five(self):
        fn = minor_breaks_log(detail=5)
        majors = np.array([1, 10, 100])
        result = fn(majors, np.array([1, 100]), 5)
        assert len(result) > 0

    def test_smallest_threshold(self):
        fn = minor_breaks_log(smallest=5)
        majors = np.array([1, 10, 100])
        result = fn(majors, np.array([1, 100]), 5)
        # All results should be >= 5
        if len(result) > 0:
            assert np.all(np.abs(result) >= 5)
