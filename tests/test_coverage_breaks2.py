"""Coverage tests for scales/breaks.py – targeting uncovered lines."""

import math
import warnings
import numpy as np
import pytest

from scales.breaks import (
    _extended,
    _pretty,
    _simplicity,
    _coverage,
    _coverage_max,
    _density,
    _density_max,
    breaks_extended,
    breaks_pretty,
    breaks_width,
    breaks_timespan,
    breaks_exp,
    cbreaks,
    extended_breaks,
    pretty_breaks,
)


# ---------------------------------------------------------------------------
# _simplicity (line 58)
# ---------------------------------------------------------------------------

class TestSimplicity:
    def test_basic(self):
        # q_idx=0, n_Q=6, j=1, lmin=0, lmax=10, step=2
        s = _simplicity(0, 6, 1, 0, 10, 2)
        assert isinstance(s, float)

    def test_zero_in_range(self):
        # When lmin <= 0 <= lmax and lmin/step is int
        s = _simplicity(0, 6, 1, 0, 10, 5)
        assert isinstance(s, float)


# ---------------------------------------------------------------------------
# _coverage edge cases (lines 67-73)
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_zero_range(self):
        result = _coverage(5, 5, 0, 10)
        assert result == 1.0

    def test_normal(self):
        result = _coverage(0, 10, 0, 10)
        assert result == 1.0


class TestCoverageMax:
    def test_span_greater(self):
        result = _coverage_max(0, 10, 20)
        assert isinstance(result, float)

    def test_span_less(self):
        result = _coverage_max(0, 10, 5)
        assert result == 1.0

    def test_zero_range(self):
        result = _coverage_max(5, 5, 0)
        assert result == 1.0


# ---------------------------------------------------------------------------
# _density edge cases (line 82)
# ---------------------------------------------------------------------------

class TestDensity:
    def test_basic(self):
        result = _density(5, 5, 0, 10, 0, 10)
        assert isinstance(result, float)

    def test_equal_lminlmax(self):
        result = _density(5, 5, 0, 10, 5, 5)
        assert isinstance(result, float)


class TestDensityMax:
    def test_k_ge_m(self):
        result = _density_max(10, 5)
        assert isinstance(result, float)

    def test_k_lt_m(self):
        result = _density_max(3, 5)
        assert result == 1.0

    def test_m_one(self):
        result = _density_max(5, 1)
        assert result == 1.0


# ---------------------------------------------------------------------------
# breaks_extended edge cases (lines 158, 168-169, 188)
# ---------------------------------------------------------------------------

class TestBreaksExtended:
    def test_very_small_range(self):
        brk = breaks_extended(n=5)
        result = brk([5.0, 5.0])
        assert len(result) == 1

    def test_with_n_override(self):
        brk = breaks_extended(n=5)
        result = brk([0, 100], n_=10)
        assert len(result) > 0

    def test_empty_finite(self):
        brk = breaks_extended(n=5)
        result = brk([float("nan"), float("inf")])
        assert len(result) == 0

    def test_only_loose(self):
        brk = breaks_extended(n=5, only_loose=True)
        result = brk([0, 100])
        assert len(result) > 0
        # Breaks should enclose data range
        assert result[0] <= 0
        assert result[-1] >= 100


# ---------------------------------------------------------------------------
# _pretty edge cases (lines 224, 238, 243)
# ---------------------------------------------------------------------------

class TestPrettyInternal:
    def test_non_finite(self):
        result = _pretty(float("nan"), 10, n=5)
        assert len(result) == 2

    def test_zero_range(self):
        result = _pretty(5, 5, n=5)
        assert len(result) == 1

    def test_small_cell(self):
        # Test when cell < 20*1e-07*max(abs(dmin), abs(dmax))
        result = _pretty(1e10, 1e10 + 1, n=5)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# breaks_pretty (lines 352)
# ---------------------------------------------------------------------------

class TestBreaksPretty:
    def test_empty(self):
        brk = breaks_pretty(n=5)
        result = brk([float("nan")])
        assert len(result) == 0

    def test_with_n_override(self):
        brk = breaks_pretty(n=5)
        result = brk([0, 100], n_=3)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# breaks_width edge cases (lines 387, 393)
# ---------------------------------------------------------------------------

class TestBreaksWidth:
    def test_negative_width(self):
        with pytest.raises(ValueError):
            breaks_width(width=-1)

    def test_with_offset(self):
        brk = breaks_width(width=5, offset=2)
        result = brk([0, 20])
        assert len(result) > 0

    def test_empty(self):
        brk = breaks_width(width=5)
        result = brk([float("nan")])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# breaks_timespan (lines 453, 460-471)
# ---------------------------------------------------------------------------

class TestBreaksTimespan:
    def test_unknown_unit(self):
        with pytest.raises(ValueError):
            breaks_timespan(unit="fortnights")

    def test_mins(self):
        brk = breaks_timespan(unit="mins", n=5)
        result = brk([0, 7200])
        assert len(result) > 0

    def test_empty(self):
        brk = breaks_timespan(unit="secs", n=5)
        result = brk([float("nan")])
        assert len(result) == 0

    def test_with_n_override(self):
        brk = breaks_timespan(unit="secs", n=5)
        result = brk([0, 3600], n_=3)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# breaks_exp (lines 508, 519, 563-578)
# ---------------------------------------------------------------------------

class TestBreaksExp:
    def test_large_range(self):
        brk = breaks_exp(n=5)
        result = brk([0.01, 1000])
        assert len(result) > 0

    def test_small_range(self):
        brk = breaks_exp(n=5)
        result = brk([1, 5])
        assert len(result) > 0

    def test_empty(self):
        brk = breaks_exp(n=5)
        result = brk([float("nan")])
        assert len(result) == 0

    def test_negative_data(self):
        brk = breaks_exp(n=5)
        result = brk([-10, -1])
        assert len(result) > 0

    def test_with_n_override(self):
        brk = breaks_exp(n=5)
        result = brk([0.1, 10000], n_=3)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# cbreaks deprecated (lines 563-578)
# ---------------------------------------------------------------------------

class TestCbreaks:
    def test_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cbreaks([0, 100])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "breaks" in result
            assert "labels" in result

    def test_with_labels_fun(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = cbreaks(
                [0, 100],
                labels_fun=lambda x: [f"{v:.1f}" for v in x]
            )
            assert all("." in lbl for lbl in result["labels"])


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

class TestLegacyAliases:
    def test_extended_breaks(self):
        assert extended_breaks is breaks_extended

    def test_pretty_breaks(self):
        assert pretty_breaks is breaks_pretty
