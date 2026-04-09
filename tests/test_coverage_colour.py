"""Coverage tests for scales/colour_mapping.py and colour_manip.py."""

import numpy as np
import pytest

from scales.colour_mapping import (
    col_numeric,
    col_bin,
    col_quantile,
    col_factor,
    _to_palette_func,
    _safe_palette_func,
    _rescale,
    _pretty_breaks,
)

from scales.colour_manip import (
    alpha,
    muted,
    col_mix,
    col_shift,
    col2hcl,
)

from scales.colour_ramp import colour_ramp


# ===========================================================================
# colour_mapping.py
# ===========================================================================

# ---------------------------------------------------------------------------
# _rescale edge (line 127)
# ---------------------------------------------------------------------------

class TestRescaleInternal:
    def test_zero_range(self):
        x = np.array([5, 5, float("nan")])
        result = _rescale(x, 5, 5)
        assert result[0] == 0.5
        assert np.isnan(result[2])


# ---------------------------------------------------------------------------
# _pretty_breaks edge (lines 140, 147, 151)
# ---------------------------------------------------------------------------

class TestPrettyBreaksInternal:
    def test_zero_step(self):
        result = _pretty_breaks(5, 5, 5)
        assert len(result) == 2

    def test_residual_ranges(self):
        # Test different residual values to cover all branches
        # residual <= 1.5
        result = _pretty_breaks(0, 10, 10)
        assert len(result) > 0

        # residual <= 3.0
        result = _pretty_breaks(0, 20, 10)
        assert len(result) > 0

        # residual <= 7.0
        result = _pretty_breaks(0, 50, 10)
        assert len(result) > 0

        # residual > 7.0
        result = _pretty_breaks(0, 100, 10)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _to_palette_func / _safe_palette_func (lines 75, 127)
# ---------------------------------------------------------------------------

class TestPaletteFunc:
    def test_string_palette(self):
        ramp = _to_palette_func("viridis")
        result = ramp(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3

    def test_nan_handling(self):
        ramp = _to_palette_func("viridis")
        result = ramp(np.array([float("nan")]))
        assert result[0] is None

    def test_safe_palette(self):
        ramp = _safe_palette_func("viridis", na_color="#808080")
        result = ramp(np.array([float("nan"), 0.5]))
        assert result[0] == "#808080"


# ---------------------------------------------------------------------------
# col_numeric edge cases (lines 206-209)
# ---------------------------------------------------------------------------

class TestColNumeric:
    def test_auto_domain(self):
        f = col_numeric("viridis", domain=None)
        result = f([0, 50, 100])
        assert len(result) == 3

    def test_all_nan(self):
        f = col_numeric("viridis", domain=None)
        result = f([float("nan"), float("nan")])
        assert all(c == "#808080" for c in result)

    def test_reverse(self):
        f = col_numeric("viridis", domain=(0, 100), reverse=True)
        result = f([0, 100])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# col_bin edge cases (lines 273-277, 282, 290, 299, 314-317)
# ---------------------------------------------------------------------------

class TestColBin:
    def test_auto_domain(self):
        f = col_bin("viridis", domain=None)
        result = f([1, 5, 10])
        assert len(result) == 3

    def test_explicit_breaks(self):
        f = col_bin("viridis", bins=[0, 5, 10])
        result = f([1, 7])
        assert len(result) == 2

    def test_not_pretty(self):
        f = col_bin("viridis", domain=(0, 100), bins=5, pretty=False)
        result = f([10, 50, 90])
        assert len(result) == 3

    def test_nan(self):
        f = col_bin("viridis", domain=(0, 10))
        result = f([float("nan"), 5])
        assert result[0] == "#808080"

    def test_right_closed(self):
        f = col_bin("viridis", domain=(0, 10), right=True)
        result = f([5])
        assert len(result) == 1

    def test_reverse(self):
        f = col_bin("viridis", domain=(0, 10), reverse=True)
        result = f([5])
        assert len(result) == 1

    def test_all_nan_auto(self):
        f = col_bin("viridis", domain=None)
        result = f([float("nan")])
        assert result[0] == "#808080"


# ---------------------------------------------------------------------------
# col_quantile edge cases (lines 374, 388-391)
# ---------------------------------------------------------------------------

class TestColQuantile:
    def test_auto_domain(self):
        f = col_quantile("viridis", domain=None, n=4)
        result = f([1, 2, 3, 4, 5])
        assert len(result) == 5

    def test_with_domain(self):
        f = col_quantile("viridis", domain=[1, 2, 3, 4, 5], n=4)
        result = f([1, 3, 5])
        assert len(result) == 3

    def test_with_probs(self):
        f = col_quantile("viridis", domain=[1, 2, 3, 4, 5],
                         probs=[0, 0.5, 1])
        result = f([1, 3, 5])
        assert len(result) == 3

    def test_all_nan(self):
        f = col_quantile("viridis", domain=None, n=4)
        result = f([float("nan")])
        assert result[0] == "#808080"


# ---------------------------------------------------------------------------
# col_factor edge cases (lines 456-457, 473, 479-483)
# ---------------------------------------------------------------------------

class TestColFactor:
    def test_auto_levels(self):
        f = col_factor("viridis", domain=None)
        result = f(["a", "b", "c"])
        assert len(result) == 3

    def test_with_levels(self):
        f = col_factor("viridis", levels=["a", "b", "c"])
        result = f(["a", "b"])
        assert len(result) == 2

    def test_reverse(self):
        f = col_factor("viridis", domain=["a", "b", "c"], reverse=True)
        result = f(["a", "b", "c"])
        assert len(result) == 3

    def test_numpy_input(self):
        f = col_factor("viridis", domain=["a", "b"])
        result = f(np.array(["a", "b"]))
        assert len(result) == 2

    def test_unknown_level(self):
        f = col_factor("viridis", domain=["a", "b"])
        result = f(["a", "z"])
        assert result[1] == "#808080"


# ===========================================================================
# colour_manip.py
# ===========================================================================

# ---------------------------------------------------------------------------
# alpha edge cases (lines 227-228, 239, 242)
# ---------------------------------------------------------------------------

class TestAlphaEdge:
    def test_none_alpha(self):
        result = alpha("red", None)
        assert isinstance(result, str)

    def test_list_colours_none_alpha(self):
        result = alpha(["red", "blue"], None)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_single_colour_broadcast(self):
        result = alpha(["red"], [0.5, 0.8])
        assert len(result) == 2

    def test_incompatible_lengths(self):
        with pytest.raises(ValueError):
            alpha(["red", "blue"], [0.5, 0.8, 0.3])

    def test_none_in_alpha_list(self):
        result = alpha(["red", "blue"], [0.5, None])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# col_mix (line 416-423)
# ---------------------------------------------------------------------------

class TestColMixEdge:
    def test_lab_space(self):
        result = col_mix("red", "blue", amount=0.5, space="lab")
        assert result.startswith("#")

    def test_rgb_space(self):
        result = col_mix("red", "blue", amount=0.5, space="rgb")
        assert result.startswith("#")


# ---------------------------------------------------------------------------
# colour_ramp edge (line 65)
# ---------------------------------------------------------------------------

class TestColourRampEdge:
    def test_too_few_colours(self):
        with pytest.raises(ValueError):
            colour_ramp(["#FF0000"])

    def test_no_alpha(self):
        ramp = colour_ramp(["#000000", "#FFFFFF"], alpha=False)
        result = ramp([0.0, 0.5, 1.0])
        assert len(result) == 3
        # Without alpha, hex should be 7 chars
        for c in result:
            assert c is not None
            assert len(c) == 7


# ---------------------------------------------------------------------------
# col2hcl edge
# ---------------------------------------------------------------------------

class TestCol2hcl:
    def test_basic(self):
        result = col2hcl("red")
        assert isinstance(result, str)
        assert result.startswith("#")

    def test_with_overrides(self):
        result = col2hcl("red", h=120, c=50, l=50)
        assert isinstance(result, str)

    def test_list_input(self):
        result = col2hcl(["red", "blue"])
        assert isinstance(result, list)
        assert len(result) == 2
