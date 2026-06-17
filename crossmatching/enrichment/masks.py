from __future__ import annotations
import numpy as np


def temperate_mask(
    pl_insol,
    pl_insol_err1,
    pl_insol_err2,
    lower: float,
    upper: float,
    use_interval: bool = False,
) -> np.ndarray:
    """Boolean mask selecting planets within [lower, upper] insolation (S_earth).

    Parameters
    ----------
    pl_insol : array-like or MaskedColumn
        Central insolation values.
    pl_insol_err1 : array-like, MaskedColumn, or None
        Upper 1-sigma uncertainty (positive magnitude). Masked/NaN → treated as 0.
    pl_insol_err2 : array-like, MaskedColumn, or None
        Lower 1-sigma uncertainty (positive magnitude). Masked/NaN → treated as 0.
    lower, upper : float
        Insolation bounds (inclusive).
    use_interval : bool
        False (default): include only planets whose central pl_insol is in range.
        True: include planets where [pl_insol − err2, pl_insol + err1] overlaps
        [lower, upper] — i.e., the planet *could* be temperate within uncertainty.
    """
    flux = np.ma.asarray(pl_insol, dtype=float)
    valid = ~np.ma.getmaskarray(flux)

    if not use_interval:
        return valid & (flux.data >= lower) & (flux.data <= upper)

    def _to_err(arr):
        if arr is None:
            return np.zeros(len(flux))
        a = np.ma.asarray(arr, dtype=float)
        bad = np.ma.getmaskarray(a) | ~np.isfinite(a.data)
        return np.where(bad, 0.0, np.abs(a.data))

    err1 = _to_err(pl_insol_err1)
    err2 = _to_err(pl_insol_err2)
    return valid & (flux.data - err2 <= upper) & (flux.data + err1 >= lower)


def rocky_mask(
    r,
    r_lower_bound,
    r_upper_bound,
    lower: float = 0.5,
    upper: float = 1.5,
    use_interval: bool = False,
) -> np.ndarray:
    """Boolean mask selecting rocky planets within [lower, upper] R_Earth.

    Parameters
    ----------
    r : array-like or MaskedColumn
        Directly measured planet radii in R_Earth (caller converts from R_Jup via R_JUP_TO_EARTH).
    r_lower_bound : array-like, MaskedColumn, or None
        Lower bound on estimated radius from msini. Masked/NaN → no bound.
    r_upper_bound : array-like, MaskedColumn, or None
        Upper bound on estimated radius from msini. Masked/NaN → no bound.
    lower, upper : float
        Radius bounds in R_Earth (inclusive, defaults: 0.5, 1.5).
    use_interval : bool
        False (default): only rows where r is directly measured and in range.
        True: also rows where r is masked but [r_lower_bound, r_upper_bound]
        overlaps [lower, upper] — i.e., the planet *could* be rocky.
    """
    r = np.ma.asarray(r, dtype=float)
    valid = ~np.ma.getmaskarray(r) & (r.data > 0)  # r=0 means absent, same as masked

    known_rocky = valid & (r.data >= lower) & (r.data <= upper)

    if not use_interval:
        return known_rocky

    def _to_bound(arr):
        if arr is None:
            return np.full(len(r), np.nan)
        a = np.ma.asarray(arr, dtype=float)
        bad = np.ma.getmaskarray(a) | ~np.isfinite(a.data)
        return np.where(bad, np.nan, a.data)

    rmin = _to_bound(r_lower_bound)
    rmax = _to_bound(r_upper_bound)
    has_bounds = ~valid & np.isfinite(rmin) & np.isfinite(rmax)
    uncertain_rocky = has_bounds & (rmin <= upper) & (rmax >= lower)

    return known_rocky | uncertain_rocky
