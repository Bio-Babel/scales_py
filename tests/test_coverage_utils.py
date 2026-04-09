"""Coverage tests for scales/_utils.py – targeting uncovered lines."""

import numpy as np
import pytest
from datetime import timedelta

from scales._utils import (
    zero_range,
    expand_range,
    rescale_common,
    recycle_common,
    fullseq,
    round_any,
    offset_by,
    precision,
)


# ---------------------------------------------------------------------------
# zero_range edge (line 68)
# ---------------------------------------------------------------------------

class TestZeroRange:
    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            zero_range([1, 2, 3])


# ---------------------------------------------------------------------------
# rescale_common (line 158-165)
# ---------------------------------------------------------------------------

class TestRescaleCommon:
    def test_zero_from_range(self):
        result = rescale_common([1, 2, 3], to=(0, 1), from_range=(5, 5))
        assert np.allclose(result, 0.5)

    def test_normal(self):
        result = rescale_common([0, 5, 10], to=(0, 1), from_range=(0, 10))
        assert np.allclose(result, [0, 0.5, 1])


# ---------------------------------------------------------------------------
# recycle_common edge (lines 197-219)
# ---------------------------------------------------------------------------

class TestRecycleCommon:
    def test_all_scalar(self):
        result = recycle_common([1], [2])
        assert len(result) == 2
        assert len(result[0]) == 1

    def test_mixed_lengths(self):
        result = recycle_common([1], [1, 2, 3])
        assert len(result[0]) == 3
        assert np.all(result[0] == 1)

    def test_incompatible(self):
        with pytest.raises(ValueError):
            recycle_common([1, 2], [1, 2, 3])

    def test_explicit_size(self):
        result = recycle_common([1], [2], size=5)
        assert len(result[0]) == 5
        assert len(result[1]) == 5


# ---------------------------------------------------------------------------
# fullseq edge (lines 248, 251, 257-258)
# ---------------------------------------------------------------------------

class TestFullseq:
    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            fullseq([1, 2, 3], size=1)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            fullseq([0, 10], size=0)

    def test_pad(self):
        result = fullseq([0, 10], size=5, pad=True)
        assert result[0] < 0
        assert result[-1] > 10


# ---------------------------------------------------------------------------
# offset_by (line 314)
# ---------------------------------------------------------------------------

class TestOffsetBy:
    def test_numeric(self):
        result = offset_by(5, 3)
        assert result == 8

    def test_datetime64(self):
        x = np.datetime64("2020-01-01")
        result = offset_by(x, np.timedelta64(1, "D"))
        assert result == np.datetime64("2020-01-02")


# ---------------------------------------------------------------------------
# precision edge (lines 350, 355)
# ---------------------------------------------------------------------------

class TestPrecision:
    def test_all_same(self):
        result = precision([5, 5, 5])
        assert result == 1.0

    def test_single_value(self):
        result = precision([5])
        assert result == 1.0

    def test_no_diffs(self):
        result = precision([float("nan"), float("inf")])
        assert result == 1.0

    def test_all_nan(self):
        result = precision([float("nan")])
        assert result == 1.0
