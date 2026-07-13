from __future__ import annotations
import numpy as np

T_SUN = 5778.0        # K

_MANN_TEFF_COEFFS = (10.5440, -33.7546, 35.1909, -11.5928)  # erratum ApJ 819 87, Table 1
_MANN_MKS_COEFFS  = (1.9515, -0.3520, 0.01680)              # erratum ApJ 819 87, Table 1
_TORRES_COEFFS    = (2.4427, 0.6679, 0.1771, 0.705, -0.21415, 0.02306, 0.04173)


def mann_teff_radius(teff: float, met: float = None) -> float:
    """Mann et al. 2015 Teff polynomial for K7–M7 dwarfs (valid ~2400–4200 K).

    Scatter: 13.4% (Teff only) or 9.3% (with [Fe/H]). Erratum: ApJ 819 87.
    """
    if not (teff > 0):
        return 0.0
    a, b, c, d = _MANN_TEFF_COEFFS
    x = teff / 3500.0
    r = a + b * x + c * x ** 2 + d * x ** 3
    if met is not None and np.isfinite(met):
        r *= 1.0 + 0.4565 * met
    return max(r, 0.05)


def mann_mks_radius(mks: float, met: float = None) -> float:
    """Mann et al. 2015 M_Ks polynomial for K7–M7 dwarfs (valid M_Ks ≈ 4–10.5).

    Scatter: 2.89% (no [Fe/H]) or 2.70% (with [Fe/H]). Erratum: ApJ 819 87.
    """
    a, b, c = _MANN_MKS_COEFFS
    r = a + b * mks + c * mks ** 2
    if met is not None and np.isfinite(met):
        r *= 1.0 - 0.04458 * met
    return max(r, 0.05)


def torres_radius(teff: float, logg: float, met: float = None) -> float:
    """Torres et al. 2010 (A&A Rev 18 67) spectroscopic calibration for FGK stars.

    Valid range: ~4700–8500 K (F0–K5). Scatter: ~3%. Requires log g.
    """
    b = _TORRES_COEFFS
    X = np.log10(teff) - 4.1
    fe_h = met if (met is not None and np.isfinite(met)) else 0.0
    log_r = (b[0] + b[1] * X + b[2] * X ** 2 + b[3] * X ** 3
             + b[4] * logg ** 2 + b[5] * logg ** 3 + b[6] * fe_h)
    return max(10.0 ** log_r, 0.05)


def _zams_exponent(teff: float, spec: str = '') -> float:
    """Spectral-class-aware ZAMS Teff→R exponent (Boyajian+ 2012, solar calibration)."""
    spec1 = spec.strip()[:1].upper() if spec else ''
    if spec1 == 'M' or (not spec1 and teff < 3900):
        return 2.5
    if spec1 == 'K' or (not spec1 and teff < 5200):
        return 2.1
    if spec1 in ('A', 'B', 'O') or (not spec1 and teff > 7500):
        return 1.0
    return 1.8


def ms_radius_from_teff(teff: float, spec: str = '') -> float:
    """Rough ZAMS Teff → radius (R/R_sun) for main-sequence fallback."""
    if not (teff > 0):
        return 0.0
    return max((teff / T_SUN) ** _zams_exponent(teff, spec), 0.05)


def mass_radius_chen_kipping(m_earth: float) -> float:
    """Chen & Kipping (2017, ApJ 834 17) mass-radius relation in Earth units.

    Piecewise power law — continuous at the Terran/Neptunian boundary (~2.04 M_Earth):
      Terran   (M < 2.04):  R = 1.008 × M^0.279
      Neptunian (2.04-132): R = 0.808 × M^0.589
    Returns 0.0 for non-positive input.
    """
    if m_earth <= 0:
        return 0.0
    if m_earth < 2.04:
        return 1.008 * m_earth ** 0.279
    elif m_earth < 132.0:
        return 0.808 * m_earth ** 0.589
    else:
        return 17.74 * m_earth ** (-0.044)
