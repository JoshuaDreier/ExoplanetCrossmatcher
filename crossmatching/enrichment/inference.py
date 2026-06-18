from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from crossmatching.enrichment.radius_estimation import (
    _mann_mks_radius,
    _mann_teff_radius,
    _torres_radius,
    _zams_exponent,
    ms_radius_from_teff,
    T_SUN,
)
from crossmatching.enrichment.spectral_types import spectype_to_teff, get_spectral_class_range


@dataclass
class ParamQty:
    """Holds a single parameter quantity with value, mask, source, and upper/lower errors."""
    val: float = 0.0
    mask: bool = True
    src: str = ""
    err1: float = np.nan
    err2: float = np.nan


@dataclass
class ParamStr:
    """Holds a single string parameter with value, mask, and provenance."""
    val: str = ""
    mask: bool = True
    src: str = ""


def _safe_err_sq(err: float, val: float = 1.0, coeff: float = 1.0) -> float:
    """Safely compute the squared error term (coeff * err / val)^2.

    Computes the scaled squared fractional error term commonly used in error
    propagation formulas, ensuring no division by zero or NaN issues occur
    if the error or value is invalid.

    Parameters
    ----------
    err : float
        The absolute error (1 sigma uncertainty).
    val : float, optional
        The measured value the error is associated with. Default is 1.0.
    coeff : float, optional
        A scaling coefficient applied to the fractional error. Default is 1.0.

    Returns
    -------
    float
        The calculated squared error term, or 0.0 if the input error is not finite.
    """
    return (coeff * err / val) ** 2 if np.isfinite(err) else 0.0


def infer_star_teff(
    teff: ParamQty,
    rad: ParamQty,
    lum: ParamQty,
    spectype: ParamStr,
) -> ParamQty:
    r"""Derive stellar effective temperature if it is masked, using physical or statistical relations.

    Calculates the temperature using the Stefan-Boltzmann law if radius and luminosity
    are available:
    
    $$ T_{\text{eff}} = T_{\odot} \left( \frac{L}{R^2} \right)^{1/4} $$

    If not, estimates the temperature from the spectral type.

    Parameters
    ----------
    teff : ParamQty
        Stellar effective temperature data.
    rad : ParamQty
        Stellar radius data.
    lum : ParamQty
        Stellar luminosity data.
    spectype : ParamStr
        Stellar spectral type data.

    Returns
    -------
    ParamQty
        The updated or original teff ParamQty.
    """
    if not teff.mask:
        return teff

    # 1. Try exact physical Stefan-Boltzmann relation first
    if not rad.mask and not lum.mask and rad.val > 0 and lum.val > 0:
        t = T_SUN * (lum.val / (rad.val**2))**0.25
        
        # Error propagation
        # \sigma_T = T\sqrt{\frac{1}{16}\left(\frac{\sigma_L}{L}\right)^2 + \frac{1}{4}\left(\frac{\sigma_R}{R}\right)^2}
        # but with asymetric errors
        err1 = t * np.sqrt(_safe_err_sq(lum.err1, lum.val, 0.25) + _safe_err_sq(rad.err2, rad.val, 0.5))
        err2 = t * np.sqrt(_safe_err_sq(lum.err2, lum.val, 0.25) + _safe_err_sq(rad.err1, rad.val, 0.5))

        return ParamQty(
            val=t,
            mask=False,
            src=f"StephanBoltzmann_derived(rad:{rad.src} lum:{lum.src})",
            err1=err1,
            err2=err2
        )
        
    # 2. Try estimation from spectral type
    elif not spectype.mask:
        t = spectype_to_teff(spectype.val)
        if not t:
            return teff
        err1 = err2 = 0.0
        t_min, t_max = get_spectral_class_range(spectype.val)
        err1 = t_max - t
        err2 = t - t_min                
        return ParamQty(
            val=t,
            mask=False,
            src=f"spectype_derived(spec:{spectype.src})",
            err1=err1,
            err2=err2
        )
    return teff


_LOG_G_SUN = 4.438 # log10(g_sun / cm s-2), IAU 2015 nominal solar values

def infer_star_radius(
    rad: ParamQty,
    mass: ParamQty,
    logg: ParamQty,
    teff: ParamQty,
    lum: ParamQty,
    met: ParamQty,
    kmag: ParamQty,
    dist: ParamQty,
    spectype: ParamStr,
) -> ParamQty:
    r"""Derive stellar radius if it is masked, using physical or statistical relations.

    Attempts to derive the radius in order of preference:
    1. Exact physical relation from mass and log(g):
       
       $$ R = 10^{0.5(\log g_\odot + \log_{10} M_* - \log g)}  = \sqrt{  \dfrac{g_{\odot}}{g}M_{*}}$$
       
    2. Exact physical relation from luminosity and Teff (Stefan-Boltzmann):
       
       $$ R = \sqrt{L} \left( \frac{T_{\odot}}{T_{\text{eff}}} \right)^2 $$
       
    3. Empirical/statistical polynomial relations based on Teff and other parameters

    Parameters
    ----------
    rad : ParamQty
        Stellar radius data.
    mass : ParamQty
        Stellar mass data.
    logg : ParamQty
        Surface gravity data.
    teff : ParamQty
        Stellar effective temperature data.
    lum : ParamQty
        Stellar luminosity data.
    met : ParamQty
        Stellar metallicity data.
    kmag : ParamQty
        2MASS K-band magnitude data.
    dist : ParamQty
        Distance data.
    spectype : ParamStr
        Stellar spectral type data.

    Returns
    -------
    ParamQty
        The updated or original rad ParamQty.
    """
    if not rad.mask:
        return rad

    # Try exact physical logg relation first
    if not logg.mask and not mass.mask:
        r = 10 ** (0.5 * (_LOG_G_SUN + np.log10(mass.val) - logg.val))
        
        # Error propagation
        t_m_up = _safe_err_sq(mass.err1, mass.val, 0.5)
        t_m_dn = _safe_err_sq(mass.err2, mass.val, 0.5)
        t_g_up = _safe_err_sq(logg.err1, coeff=0.5 * np.log(10.0))
        t_g_dn = _safe_err_sq(logg.err2, coeff=0.5 * np.log(10.0))
        
        err1 = r * np.sqrt(t_m_up + t_g_dn) 
        err2 = r * np.sqrt(t_m_dn + t_g_up) 
        return ParamQty(
            val=r,
            mask=False,
            src=f"logg_derived(mass:{mass.src} logg:{logg.src})",
            err1=err1,
            err2=err2
        )
        
    # Try exact physical Stefan-Boltzmann relation next
    elif not lum.mask and not teff.mask and lum.val > 0 and teff.val > 0:
        # R = \sqrt{L}\left(\frac{T_*}{T}\right)^2
        r = np.sqrt(lum.val) * (T_SUN / teff.val) ** 2
        
        # Error propagation
        # \sigma_R = R\sqrt{\frac{1}{4}\left(\frac{\sigma_L}{L}\right)^2 + 4\left(\frac{\sigma_T}{T}\right)^2}
        t_l_up = _safe_err_sq(lum.err1, lum.val, 0.5)
        t_l_dn = _safe_err_sq(lum.err2, lum.val, 0.5)
        t_t_up = _safe_err_sq(teff.err1, teff.val, 2.0)
        t_t_dn = _safe_err_sq(teff.err2, teff.val, 2.0)
        
        err1 = r * np.sqrt(t_l_up + t_t_dn)
        err2 = r * np.sqrt(t_l_dn + t_t_up)
        return ParamQty(
            val=r,
            mask=False,
            src=f"StephanBoltzmann_derived(lum:{lum.src} teff:{teff.src})",
            err1=err1,
            err2=err2
        )

    # Try empirical/statistical estimations from teff
    elif not teff.mask:
        teff_val = teff.val
        logg_val = logg.val if not logg.mask else None
        met_val  = met.val if not met.mask else None
        kmag_val = kmag.val if not kmag.mask else None
        dist_val = dist.val if not dist.mask else None

        r = 0.0
        err1 = err2 = 0
        r_tag = ''

        # 1. Mann 2015 M_Ks polynomial — best for K7–M7 (~3% scatter)
        if kmag_val is not None and dist_val is not None and dist_val > 0 and teff_val < 4200:
            mks = kmag_val - 5.0 * np.log10(dist_val / 10.0)
            if 4.0 <= mks <= 10.5:
                r = _mann_mks_radius(mks, met_val)
                r_tag = f"mann_mks(kmag:{kmag.src})"
                err1 = err2 = 0.029 * r

        # 2. Torres 2010 — best for FGK with log g (~3% scatter)
        if r == 0.0 and logg_val is not None and 3900 < teff_val < 8500:
            r = _torres_radius(teff_val, logg_val, met_val)
            r_tag = f"torres(teff:{teff.src} logg:{logg.src})"
            err1 = err2 = 0.030 * r

        # 3. Mann 2015 Teff polynomial — better than power law for M dwarfs
        if r == 0.0 and teff_val < 4000:
            r = _mann_teff_radius(teff_val, met_val)
            r_tag = f"mann_teff(teff:{teff.src})"
            frac = 0.093 if (met_val is not None and np.isfinite(met_val)) else 0.134
            err1 = err2 = frac * r

        # 4. ZAMS power‑law — last resort
        if r == 0.0:
            r = ms_radius_from_teff(teff_val, spectype.val)
            r_tag = f"ms(teff:{teff.src})"
            exp = _zams_exponent(teff_val, spectype.val)
            coeff = exp * r / teff_val
            if np.isfinite(teff.err1):
                err1 = coeff * teff.err1
            if np.isfinite(teff.err2):
                err2 = coeff * teff.err2

        if r > 0:
            return ParamQty(
                val=r,
                mask=False,
                src=r_tag,
                err1=err1,
                err2=err2
            )

    return rad


def infer_star_mass(
    mass: ParamQty,
    rad: ParamQty,
    logg: ParamQty,
) -> ParamQty:
    r"""Derive stellar mass if it is masked, using the physical log(g) relation.

    Calculates the mass from the stellar radius and surface gravity log(g):
    
    $$ M = 10^{\log g - \log g_\odot + 2 \log_{10} R} $$

    Passes thorugh unchanged if that is not possible
    Parameters
    ----------
    mass : ParamQty
        Stellar mass data.
    rad : ParamQty
        Stellar radius data.
    logg : ParamQty
        Surface gravity data.

    Returns
    -------
    ParamQty
        The updated or original mass ParamQty.
    """
    if not mass.mask:
        return mass

    if not logg.mask and not rad.mask:

        # M_* = \frac{g}{g_\odot} R^2 = 10^{\log g - \log g_\odot + 2\log R}
        m = 10 ** (logg.val - _LOG_G_SUN + 2 * np.log10(rad.val))
        # Error propagation
        #\sigma_M = \sqrt{(M_* \ln 10 \cdot \sigma_{\log g})^2 + (2M_* \frac{\sigma_R}{R})^2}
        t_r_up = _safe_err_sq(rad.err1, rad.val, 2.0)
        t_r_dn = _safe_err_sq(rad.err2, rad.val, 2.0)
        t_g_up = _safe_err_sq(logg.err1, coeff=np.log(10.0))
        t_g_dn = _safe_err_sq(logg.err2, coeff=np.log(10.0))
        
        err1 = m * np.sqrt(t_r_up + t_g_up)
        err2 = m * np.sqrt(t_r_dn + t_g_dn)
        return ParamQty(
            val=m,
            mask=False,
            src=f"logg_derived(rad:{rad.src} logg:{logg.src})",
            err1=err1,
            err2=err2
        )

    return mass
