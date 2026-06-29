from __future__ import annotations
import numpy as np


def _to_err(arr, n=1):
    if arr is None:
        return np.zeros(n) if n > 1 else 0.0
    a = np.ma.asarray(arr, dtype=float)
    bad = np.ma.getmaskarray(a) | ~np.isfinite(a.data)
    return np.where(bad, 0.0, np.abs(a.data))


def temperate_mask(
    pl_insol,
    pl_insol_max,
    pl_insol_min,
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
    
    err1 = _to_err(pl_insol_max, len(flux))
    err2 = _to_err(pl_insol_min, len(flux))
    return valid & (flux.data - err2 <= upper) & (flux.data + err1 >= lower)


def rocky_mask(
    r,
    r_min,
    r_max,
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
    r_max : array-like, MaskedColumn, or None
        Upper 1-sigma uncertainty in R_Earth (positive magnitude). Masked/NaN → treated as 0.
    r_min :  array-like, MaskedColumn, or None
        Lower 1-sigma uncertainty  in R_Earth (positive magnitude). Masked/NaN → treated as 0.
    r_lower_bound : array-like, MaskedColumn, or None
        Lower bound on estimated radius from msini in R_Earth. Masked/NaN → no bound.
    r_upper_bound : array-like, MaskedColumn, or None
        Upper bound on estimated radius from msini  in R_Earth. Masked/NaN → no bound.
    lower, upper : float
        Radius bounds in R_Earth (inclusive, defaults: 0.5, 1.5).
    use_interval : bool
        False (default): only rows where r is directly measured and in range.
        True: also rows where r is masked but [r_lower_bound, r_upper_bound]
        overlaps [lower, upper] — i.e., the planet *could* be rocky.
    """
    r = np.ma.asanyarray(r, dtype=float)
    def _to_bound(bound_arr, mean, uncertainty, sign):
        if bound_arr is None:
            return np.full(len(r), np.nan)
        bound_arr = np.ma.asanyarray(bound_arr)   
        mean_arr = np.ma.asanyarray(mean)  
        mean_arr.mask = mean_arr.mask & ~(mean_arr > 0)
        uncertainty_arr = np.ma.asanyarray(uncertainty) if uncertainty is not None else np.zeros_like(mean_arr)
        bound_arr[~mean_arr.mask] = (mean_arr + sign*uncertainty_arr.filled(0))[~mean_arr.mask]
        return bound_arr

    rmin = _to_bound(r_lower_bound, r, r_min, -1)
    rmax = _to_bound(r_upper_bound, r, r_max, 1)

    r_in_bounds = np.ma.filled((r >= lower) & (r <= upper), False)
    rmin_in_bounds = np.ma.filled((rmin >= lower) & (rmin <= upper), False)
    rmax_in_bounds = np.ma.filled((rmax >= lower) & (rmax <= upper), False)

    certain = r_in_bounds | (rmin_in_bounds & rmax_in_bounds)
    uncertain = rmin_in_bounds | rmax_in_bounds
    return (certain | uncertain) if use_interval else certain
