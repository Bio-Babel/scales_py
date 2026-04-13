"""
Colour ramp interpolation for the scales package.

Python port of R/colour-ramp.R from the R scales package
(https://github.com/r-lib/scales).  Creates callable colour ramps that
map values in [0, 1] to hex colour strings by interpolation in CIELAB
colour space, matching R's ``farver``-based implementation.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from matplotlib.colors import to_hex, to_rgba
from numpy.typing import ArrayLike

from .colour_manip import _lab_to_rgb, _rgb_to_lab
from .palettes import ContinuousPalette

__all__ = [
    "colour_ramp",
]


def colour_ramp(
    colors: Sequence[str],
    na_color: Optional[str] = None,
    alpha: bool = True,
) -> ContinuousPalette:
    """
    Create a colour ramp that maps [0, 1] values to colours.

    Interpolation is performed in CIELAB colour space by linearly
    interpolating L, a, b channels independently (matching R's
    ``farver``-based ``colour_ramp``).

    Parameters
    ----------
    colors : sequence of str
        One or more colours (any format accepted by matplotlib) that
        define the endpoints and optional interior knots of the ramp.
    na_color : str, optional
        Colour to use for ``NaN`` / missing values.  If *None*, ``NaN``
        inputs produce ``None`` entries in the output list.
    alpha : bool, default True
        If *True*, the alpha channel is preserved in the output hex strings
        (``#RRGGBBAA``).  If *False*, the alpha channel is stripped.

    Returns
    -------
    ContinuousPalette
        A palette ``f(x)`` where *x* is an array-like of floats in
        [0, 1].  Returns a list of hex colour strings of the same length.

    Examples
    --------
    >>> ramp = colour_ramp(["red", "blue"])
    >>> ramp([0.0, 0.5, 1.0])  # doctest: +SKIP
    ['#ff0000ff', '#ca2883ff', '#0000ffff']
    """
    if len(colors) == 0:
        raise ValueError("colour_ramp requires at least one colour.")

    # Single colour: return constant function (matches R behaviour)
    if len(colors) == 1:
        single = to_hex(to_rgba(colors[0]), keep_alpha=True)
        single_no_alpha = to_hex(to_rgba(colors[0])[:3], keep_alpha=False)

        def _const_ramp(x: ArrayLike) -> List[Optional[str]]:
            x = np.asarray(x, dtype=float)
            result: List[Optional[str]] = []
            for val in x.flat:
                if np.isnan(val):
                    result.append(na_color)
                else:
                    result.append(single if alpha else single_no_alpha)
            return result

        # R: new_continuous_palette(fun, type="colour", na_safe=!is.na(na.color))
        return ContinuousPalette(
            _const_ramp, type="colour", na_safe=na_color is not None
        )

    # Convert all colours to CIELAB + alpha
    rgba_list = [to_rgba(c) for c in colors]
    lab_array = np.array(
        [_rgb_to_lab(r, g, b) for r, g, b, _a in rgba_list]
    )  # shape (n, 3): L, a, b
    alpha_array = np.array([a for _r, _g, _b, a in rgba_list])

    x_in = np.linspace(0.0, 1.0, len(colors))

    # R: if (!alpha || all(lab_in[, 4] == 1)) alpha_interp returns NULL
    # When alpha is disabled or all inputs are fully opaque, skip alpha
    # interpolation and return #RRGGBB (not #RRGGBBAA).
    _has_alpha = alpha and not np.all(alpha_array == 1.0)

    def _ramp(x: ArrayLike) -> List[Optional[str]]:
        x = np.asarray(x, dtype=float)
        result: List[Optional[str]] = []
        for val in x.flat:
            if np.isnan(val) or val < 0.0 or val > 1.0:
                # R: approxfun(rule=1) returns NA for out-of-[0,1],
                # then encode_colour returns NA, mapped to na_color.
                result.append(na_color)
            else:
                L = float(np.interp(val, x_in, lab_array[:, 0]))
                a = float(np.interp(val, x_in, lab_array[:, 1]))
                b = float(np.interp(val, x_in, lab_array[:, 2]))
                r, g, bl = _lab_to_rgb(L, a, b)
                if _has_alpha:
                    a_val = float(np.interp(val, x_in, alpha_array))
                    result.append(
                        to_hex((r, g, bl, a_val), keep_alpha=True)
                    )
                else:
                    result.append(
                        to_hex((r, g, bl), keep_alpha=False)
                    )
        return result

    # R: new_continuous_palette(fun, type="colour", na_safe=!is.na(na.color))
    return ContinuousPalette(
        _ramp, type="colour", na_safe=na_color is not None
    )
