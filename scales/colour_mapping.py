"""
Colour mapping functions for the scales package.

Python port of R/colour-mapping.R from the R scales package
(https://github.com/r-lib/scales).  Provides factory functions that return
callables mapping data values to hex colour strings:

- :func:`col_numeric` -- continuous linear interpolation
- :func:`col_bin` -- binned (stepped) colour mapping
- :func:`col_quantile` -- quantile-based binning
- :func:`col_factor` -- categorical / factor mapping
"""

from __future__ import annotations

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike

from .colour_ramp import colour_ramp
from .palettes import pal_brewer, pal_viridis

__all__ = [
    "col_numeric",
    "col_bin",
    "col_quantile",
    "col_factor",
]

# ---------------------------------------------------------------------------
# Brewer palette name lookup (from embedded data)
# ---------------------------------------------------------------------------

from ._palettes_data import BREWER_MAXCOLORS as _BREWER_PALETTES

_VIRIDIS_NAMES = {"viridis", "magma", "inferno", "plasma"}


# ---------------------------------------------------------------------------
# Internal helpers (matching R's safePaletteFunc / toPaletteFunc dispatch)
# ---------------------------------------------------------------------------

def _to_palette_func(
    pal: Union[str, Sequence[str], Callable],
    alpha: bool = True,
    nlevels: Optional[int] = None,
) -> Callable[[ArrayLike], List[str]]:
    """
    Convert a palette specification to a callable ramp over [0, 1].

    Matches R's ``toPaletteFunc`` S3 dispatch:
    - A single string that is a Brewer palette name → sample colours, pass
      through ``colour_ramp`` (LAB interpolation).
    - A single string that is a viridis option → sample 256 colours, pass
      through ``colour_ramp``.
    - A character vector of colours → pass through ``colour_ramp``.
    - A callable → use as-is.

    Parameters
    ----------
    pal : str, sequence of str, or callable
        Palette specification.
    alpha : bool
        Whether to include alpha channel in interpolation.
    nlevels : int or None
        Number of levels (used for Brewer palette sampling).
    """
    if callable(pal) and not isinstance(pal, (str, list, tuple)):
        return pal

    if isinstance(pal, str):
        if pal in _BREWER_PALETTES:
            # R: sample all maxcolors or abs(nlevels) colours
            max_n = _BREWER_PALETTES[pal]
            if nlevels is not None:
                n_sample = min(abs(nlevels), max_n)
            else:
                n_sample = max_n
            colors = pal_brewer(palette=pal)(n_sample)
            colors = [c for c in colors if c is not None]
            return colour_ramp(colors, alpha=alpha)
        elif pal in _VIRIDIS_NAMES:
            colors = pal_viridis(option=pal)(256)
            return colour_ramp(colors, alpha=alpha)
        else:
            # Try as a single colour string
            pal = [pal]

    # List of colours
    return colour_ramp(list(pal), alpha=alpha)


def _safe_palette_func(
    pal: Union[str, Sequence[str], Callable],
    na_color: str,
    alpha: bool = True,
    nlevels: Optional[int] = None,
) -> Callable[[ArrayLike], List[str]]:
    """
    Wrap a palette function with NA handling and range filtering.

    Matches R's ``safePaletteFunc``: composes filterRange → filterNA →
    filterZeroLength → filterRGB → toPaletteFunc.
    """
    ramp = _to_palette_func(pal, alpha=alpha, nlevels=nlevels)

    def _safe(x: ArrayLike) -> List[str]:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return []
        # filterRange: out-of-[0,1] → NaN
        x = np.where((x < 0) | (x > 1), np.nan, x)
        result = ramp(x)
        return [na_color if c is None else c for c in result]

    return _safe


def _rescale(
    x: np.ndarray,
    domain_min: float,
    domain_max: float,
) -> np.ndarray:
    """Linearly rescale *x* from *[domain_min, domain_max]* to [0, 1]."""
    rng = domain_max - domain_min
    if rng == 0:
        return np.where(np.isnan(x), np.nan, 0.5)
    return (x - domain_min) / rng


def _pretty_breaks(domain_min: float, domain_max: float, n: int) -> np.ndarray:
    """
    Compute *n* + 1 "pretty" breakpoints spanning the domain.

    Uses numpy's ``linspace`` rounded to a clean step size that resembles
    R's ``pretty()`` heuristic.
    """
    raw_step = (domain_max - domain_min) / n
    if raw_step == 0:
        return np.array([domain_min, domain_max])

    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1.5:
        nice_step = 1.0 * magnitude
    elif residual <= 3.0:
        nice_step = 2.0 * magnitude
    elif residual <= 7.0:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude

    lo = np.floor(domain_min / nice_step) * nice_step
    hi = np.ceil(domain_max / nice_step) * nice_step
    return np.arange(lo, hi + nice_step * 0.5, nice_step)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def col_numeric(
    palette: Union[str, Sequence[str]],
    domain: Optional[Tuple[float, float]] = None,
    na_color: str = "#808080",
    alpha: bool = False,
    reverse: bool = False,
) -> Callable[[ArrayLike], List[str]]:
    """
    Map continuous numeric values to colours via linear interpolation.

    Parameters
    ----------
    palette : str or sequence of str
        Colourmap name (e.g. ``"Blues"``, ``"Greens"``, ``"viridis"``) or
        a list of colour strings defining the ramp.
    domain : tuple of (float, float), optional
        ``(min, max)`` of the data domain.  If *None*, the domain is
        inferred from the first call.
    na_color : str, default "#808080"
        Colour for missing / ``NaN`` values.
    alpha : bool, default False
        If *True*, alpha channels in the palette colours are included
        in interpolation and output.
    reverse : bool, default False
        Reverse the palette direction.

    Returns
    -------
    callable
        ``f(x)`` mapping numeric array to a list of hex colour strings.

    Examples
    --------
    >>> f = col_numeric(["white", "red"], domain=(0, 100))
    >>> f([0, 50, 100])  # doctest: +SKIP
    ['#ffffffff', '#ff8080ff', '#ff0000ff']
    """
    ramp = _safe_palette_func(palette, na_color, alpha=alpha)

    # Mutable state for auto-domain
    state: Dict[str, Any] = {
        "domain": domain,
    }

    def _map(x: ArrayLike) -> List[str]:
        x = np.asarray(x, dtype=float)

        if state["domain"] is None:
            finite = x[np.isfinite(x)]
            if len(finite) == 0:
                return [na_color] * x.size
            state["domain"] = (float(finite.min()), float(finite.max()))

        lo, hi = state["domain"]
        scaled = _rescale(x, lo, hi)
        if reverse:
            scaled = 1.0 - scaled
        # R: warn when values are outside the color scale
        if np.any((scaled < 0) | (scaled > 1), where=~np.isnan(scaled)):
            warnings.warn(
                "Some values were outside the color scale and will be "
                "treated as NA",
                stacklevel=2,
            )
        return ramp(scaled)

    return _map


def col_bin(
    palette: Union[str, Sequence[str]],
    domain: Optional[Tuple[float, float]] = None,
    bins: Union[int, Sequence[float]] = 7,
    pretty: bool = True,
    na_color: str = "#808080",
    alpha: bool = False,
    reverse: bool = False,
    right: bool = False,
) -> Callable[[ArrayLike], List[str]]:
    """
    Map continuous data to colours through binning.

    Parameters
    ----------
    palette : str or sequence of str
        Palette specification.
    domain : tuple of (float, float), optional
        Data domain.  Required if *bins* is an integer and *pretty* is False.
    bins : int or sequence of float, default 7
        Number of bins or explicit breakpoints.
    pretty : bool, default True
        Use "pretty" breakpoints when *bins* is an integer.
    na_color : str, default "#808080"
        Colour for missing values.
    alpha : bool, default False
        If *True*, preserve alpha channels in interpolation.
    reverse : bool, default False
        Reverse palette direction.
    right : bool, default False
        If *True*, bins are right-closed ``(a, b]``; otherwise left-closed
        ``[a, b)``.

    Returns
    -------
    callable
        ``f(x)`` mapping numeric array to a list of hex colour strings.
    """
    # R: autobin = is.null(domain) && length(bins) == 1
    autobin = domain is None and isinstance(bins, int)

    state: Dict[str, Any] = {
        "breaks": None,
    }

    if not isinstance(bins, int) and domain is None:
        # R: explicit breaks don't need a domain
        state["breaks"] = np.sort(np.asarray(bins, dtype=float))
    elif domain is not None:
        state["breaks"] = _get_bins(domain, None, bins, pretty)

    def _map(x: ArrayLike) -> List[str]:
        x = np.atleast_1d(np.asarray(x, dtype=float))

        if x.size == 0 or np.all(np.isnan(x)):
            return [na_color] * x.size

        breaks = state["breaks"]
        if breaks is None:
            breaks = _get_bins(None, x, bins, pretty)
            state["breaks"] = breaks

        n_bins = len(breaks) - 1
        if n_bins < 1:
            return [na_color] * x.size

        # R: col_bin delegates to col_factor(palette, domain=1:numColors, ...)
        # This creates a discrete color mapping with exactly numColors colors.
        color_func = col_factor(
            palette,
            domain=[str(i) for i in range(1, n_bins + 1)],
            na_color=na_color,
            alpha=alpha,
            reverse=reverse,
        )

        # R: cut(x, breaks, labels=FALSE, include.lowest=TRUE, right=right)
        ints = _cut(x, breaks, include_lowest=True, right=right)

        # Map bin labels through col_factor
        labels = [str(int(v)) if not np.isnan(v) else "NA" for v in ints]
        result: List[str] = []
        for lab, val in zip(labels, x.flat):
            if np.isnan(val) or lab == "NA":
                result.append(na_color)
            else:
                mapped = color_func([lab])
                result.append(mapped[0])

        return result

    return _map


def _get_bins(
    domain: Optional[Tuple[float, float]],
    x: Optional[np.ndarray],
    bins: Union[int, Sequence[float]],
    pretty: bool,
) -> np.ndarray:
    """Compute bin breakpoints (R's ``getBins``)."""
    if not isinstance(bins, int):
        return np.sort(np.asarray(bins, dtype=float))
    if bins < 2:
        raise ValueError(f"Invalid bins value ({bins}); bin count must be at least 2")

    if domain is not None:
        ref = np.asarray(domain, dtype=float)
    elif x is not None:
        ref = x[np.isfinite(x)]
    else:
        raise ValueError("domain and x can't both be None")

    if len(ref) == 0:
        return np.array([0.0, 1.0])

    if pretty:
        return _pretty_breaks(float(ref.min()), float(ref.max()), bins)
    else:
        return np.linspace(float(ref.min()), float(ref.max()), bins + 1)


def _cut(
    x: np.ndarray,
    breaks: np.ndarray,
    include_lowest: bool = True,
    right: bool = False,
) -> np.ndarray:
    """
    Bin values into integer labels (R's ``cut(..., labels=FALSE)``).

    Returns 1-based bin indices; NaN for values outside the breaks range.
    """
    n_bins = len(breaks) - 1
    result = np.full(x.shape, np.nan, dtype=float)

    for i in range(x.size):
        val = x.flat[i]
        if np.isnan(val):
            continue

        assigned = False
        for b in range(n_bins):
            lo, hi = breaks[b], breaks[b + 1]

            if right:
                # (lo, hi] — right-closed
                in_bin = (val > lo) and (val <= hi)
                # include.lowest: first bin becomes [lo, hi]
                if include_lowest and b == 0:
                    in_bin = (val >= lo) and (val <= hi)
            else:
                # [lo, hi) — left-closed
                in_bin = (val >= lo) and (val < hi)
                # include.lowest: last bin becomes [lo, hi]
                if include_lowest and b == n_bins - 1:
                    in_bin = (val >= lo) and (val <= hi)

            if in_bin:
                result.flat[i] = b + 1  # 1-based
                assigned = True
                break

        # not assigned → outside range → stays NaN

    return result


def _safe_quantile(
    x: np.ndarray,
    probs: np.ndarray,
    n_requested: int,
) -> np.ndarray:
    """R's ``safe_quantile``: deduplicate and warn on skewed data."""
    bins = np.unique(np.quantile(x, probs))
    if len(bins) < len(probs):
        warnings.warn(
            f"Skewed data means we can only allocate {len(bins)} unique "
            f"colours not the {n_requested} requested",
            stacklevel=3,
        )
    return bins


def col_quantile(
    palette: Union[str, Sequence[str]],
    domain: Optional[ArrayLike] = None,
    n: int = 4,
    probs: Optional[Sequence[float]] = None,
    na_color: str = "#808080",
    alpha: bool = False,
    reverse: bool = False,
    right: bool = False,
) -> Callable[[ArrayLike], List[str]]:
    """
    Map quantile-based bins to colours.

    Parameters
    ----------
    palette : str or sequence of str
        Palette specification.
    domain : array-like, optional
        Reference data from which quantiles are computed.  If *None*,
        quantiles are computed on the first call.
    n : int, default 4
        Number of quantile bins (ignored if *probs* is given).
    probs : sequence of float, optional
        Explicit quantile probabilities (e.g. ``[0, 0.25, 0.5, 0.75, 1]``).
    na_color : str, default "#808080"
        Colour for missing values.
    reverse : bool, default False
        Reverse palette direction.
    right : bool, default False
        Bin closure direction.

    Returns
    -------
    callable
        ``f(x)`` mapping numeric array to a list of hex colour strings.
    """
    if probs is None:
        probs_arr = np.linspace(0, 1, n + 1)
    else:
        probs_arr = np.asarray(probs, dtype=float)

    state: Dict[str, Any] = {"breaks": None}

    n_requested = len(probs_arr) - 1

    if domain is not None:
        domain_arr = np.asarray(domain, dtype=float)
        finite = domain_arr[np.isfinite(domain_arr)]
        if len(finite) > 0:
            state["breaks"] = _safe_quantile(finite, probs_arr, n_requested)

    def _map(x: ArrayLike) -> List[str]:
        x = np.asarray(x, dtype=float)

        if state["breaks"] is None:
            finite = x[np.isfinite(x)]
            if len(finite) == 0:
                return [na_color] * x.size
            state["breaks"] = _safe_quantile(finite, probs_arr, n_requested)

        # Delegate to col_bin with the computed breaks
        mapper = col_bin(
            palette,
            bins=state["breaks"],
            na_color=na_color,
            alpha=alpha,
            reverse=reverse,
            right=right,
        )
        return mapper(x)

    return _map


def col_factor(
    palette: Union[str, Sequence[str]],
    domain: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[str]] = None,
    ordered: bool = False,
    na_color: str = "#808080",
    alpha: bool = False,
    reverse: bool = False,
) -> Callable[[Union[ArrayLike, Sequence[str]]], List[str]]:
    """
    Map categorical (factor) values to colours.

    Parameters
    ----------
    palette : str or sequence of str
        Palette specification.  When a list of colours, the number of
        colours should ideally match the number of levels.
    domain : sequence of str, optional
        Valid category labels.  If *None*, inferred on first call.
    levels : sequence of str, optional
        Synonym for *domain* (mirrors R's ``levels`` argument).
    ordered : bool, default False
        If *True*, treat categories as ordered and interpolate; otherwise
        assign evenly spaced colours.
    na_color : str, default "#808080"
        Colour for missing / unknown levels.
    reverse : bool, default False
        Reverse palette direction.

    Returns
    -------
    callable
        ``f(x)`` mapping an array of category labels to a list of hex
        colour strings.
    """
    lvls = list(levels) if levels is not None else (
        list(domain) if domain is not None else None
    )

    state: Dict[str, Any] = {"levels": lvls, "colors": None}

    def _ensure_colors(x_levels: List[str]) -> Dict[str, str]:
        if state["colors"] is not None:
            return state["colors"]

        all_levels = state["levels"] if state["levels"] is not None else x_levels
        if reverse:
            all_levels = list(reversed(all_levels))

        n = len(all_levels)
        if n == 0:
            state["colors"] = {}
            return state["colors"]

        # R: safePaletteFunc(palette, na.color, alpha,
        #        nlevels = length(lvls) * ifelse(reverse, -1, 1))
        nlevels = n * (-1 if reverse else 1)
        ramp = _safe_palette_func(
            palette, na_color, alpha=alpha, nlevels=nlevels
        )

        if n == 1:
            positions = np.array([0.5])
        else:
            # R: rescale(as.integer(x), from = c(1, length(lvls)))
            positions = np.linspace(0, 1, n)

        hex_colors = ramp(positions)
        state["colors"] = dict(zip(all_levels, hex_colors))
        state["levels"] = all_levels
        return state["colors"]

    def _map(x: Union[ArrayLike, Sequence[str]]) -> List[str]:
        if isinstance(x, np.ndarray):
            labels = x.astype(str).tolist()
        else:
            labels = [str(v) for v in x]

        if state["levels"] is None:
            # R: calcLevels — ordered preserves insertion order,
            # unordered sorts alphabetically.
            seen: Dict[str, None] = {}
            for lab in labels:
                if lab not in seen:
                    seen[lab] = None
            discovered = list(seen.keys())
            if not ordered:
                discovered = sorted(discovered)
            state["levels"] = discovered

        color_map = _ensure_colors(labels)
        return [color_map.get(lab, na_color) for lab in labels]

    return _map
