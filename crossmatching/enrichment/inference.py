from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import astropy.units as u

from crossmatching.enrichment.radius_estimation import (
    _mann_mks_radius,
    _mann_teff_radius,
    _torres_radius,
    _zams_exponent,
    ms_radius_from_teff,
    mass_radius_chen_kipping,
    T_SUN,
)
from crossmatching.enrichment.spectral_types import (
    spectype_to_teff,
    get_spectral_class_range,
    spectype_display,
)


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


def _is_valid_qty(q: ParamQty) -> bool:
    """Return True if a ParamQty contains a usable finite value."""
    return not q.mask and np.isfinite(q.val)



def _masked_qty(src: str = "") -> ParamQty:
    """Return a masked ParamQty with NaN value and uncertainties."""
    return ParamQty(
        val=np.nan,
        mask=True,
        src=src,
        err1=np.nan,
        err2=np.nan,
    )


def _safe_err_sq(err: float, val: float = 1.0, coeff: float = 1.0) -> float:
    """Safely compute the squared error term (coeff * err / val)^2.

    Computes the scaled squared fractional error term commonly used in error
    propagation formulas, ensuring no division by zero or NaN issues occur
    if the error or value is invalid.

    Parameters
    ----------
    err : float
        The absolute error, usually a 1 sigma uncertainty.
    val : float, optional
        The measured value the error is associated with. Default is 1.0.
    coeff : float, optional
        A scaling coefficient applied to the fractional error. Default is 1.0.

    Returns
    -------
    float
        The calculated squared error term, or 0.0 if the input error is not finite.
    """
    return (coeff * err / val) ** 2 if np.isfinite(err) and np.isfinite(val) and val != 0 else 0.0



def _err_note(q: ParamQty) -> str:
    """Annotate missing uncertainty components for provenance strings."""
    no_err1 = not np.isfinite(q.err1)
    no_err2 = not np.isfinite(q.err2)

    if no_err1 and no_err2:
        return "[no uncert.]"
    if no_err1:
        return "[no err1]"
    if no_err2:
        return "[no err2]"
    return ""


def _src(label: str, q: ParamQty, fallback: str = "") -> str:
    """Standardized source formatter with missing-uncertainty annotation."""
    return f"{label}:{q.src or fallback}{_err_note(q)}"



def infer_star_teff(
    teff: ParamQty,
    rad: ParamQty,
    lum: ParamQty,
    spectype: ParamStr,
) -> ParamQty:
    r"""Derive stellar effective temperature if it is masked.

    Uses the Stefan-Boltzmann relation when radius and luminosity are available:

    T_eff = T_sun * (L / R^2)^(1/4)

    If that is not possible, estimates the temperature from the spectral type.

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
    if _is_valid_qty(teff):
        return teff

    # 1. Try exact physical Stefan-Boltzmann relation first
    if _is_valid_qty(rad) and _is_valid_qty(lum) and rad.val > 0 and lum.val > 0:
        t = T_SUN * (lum.val / (rad.val**2))**0.25

        # Error propagation
        # sigma_T = T * sqrt((1/16) * (sigma_L / L)^2 + (1/4) * (sigma_R / R)^2)
        # but with asymmetric errors
        err1 = t * np.sqrt(
            _safe_err_sq(lum.err1, lum.val, 0.25)
            + _safe_err_sq(rad.err2, rad.val, 0.5)
        )
        err2 = t * np.sqrt(
            _safe_err_sq(lum.err2, lum.val, 0.25)
            + _safe_err_sq(rad.err1, rad.val, 0.5)
        )

        return ParamQty(
            val=t,
            mask=False,
            src=f"StephanBoltzmann_derived({_src('rad', rad)} {_src('lum', lum)})",
            err1=err1,
            err2=err2,
        )

    # 2. Try estimation from spectral type
    elif not spectype.mask:
        t = spectype_to_teff(spectype.val)

        if not t:
            return teff

        t_min, t_max = get_spectral_class_range(spectype.val)
        err1 = t_max - t
        err2 = t - t_min

        return ParamQty(
            val=t,
            mask=False,
            src=f"spectype_derived(spec:{spectype.src})",
            err1=err1,
            err2=err2,
        )

    return teff


_LOG_G_SUN = 4.438  # log10(g_sun / cm s-2), IAU 2015 nominal solar values


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
    r"""Derive stellar radius if it is masked.

    Attempts to derive the radius in order of preference:

    1. Exact physical relation from mass and log(g).
    2. Exact physical relation from luminosity and Teff.
    3. Empirical/statistical polynomial relations based on Teff and other parameters.

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
    if _is_valid_qty(rad):
        return rad

    # Try exact physical logg relation first
    if _is_valid_qty(logg) and _is_valid_qty(mass) and mass.val > 0:
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
            src=f"logg_derived({_src('mass', mass)} {_src('logg', logg)})",
            err1=err1,
            err2=err2,
        )

    # Try exact physical Stefan-Boltzmann relation next
    elif _is_valid_qty(lum) and _is_valid_qty(teff) and lum.val > 0 and teff.val > 0:
        # R = sqrt(L) * (T_sun / T)^2
        r = np.sqrt(lum.val) * (T_SUN / teff.val) ** 2

        # Error propagation
        # sigma_R = R * sqrt((1/4) * (sigma_L / L)^2 + 4 * (sigma_T / T)^2)
        t_l_up = _safe_err_sq(lum.err1, lum.val, 0.5)
        t_l_dn = _safe_err_sq(lum.err2, lum.val, 0.5)
        t_t_up = _safe_err_sq(teff.err1, teff.val, 2.0)
        t_t_dn = _safe_err_sq(teff.err2, teff.val, 2.0)

        err1 = r * np.sqrt(t_l_up + t_t_dn)
        err2 = r * np.sqrt(t_l_dn + t_t_up)

        return ParamQty(
            val=r,
            mask=False,
            src=f"StephanBoltzmann_derived({_src('lum', lum)} {_src('teff', teff)})",
            err1=err1,
            err2=err2,
        )

    # Try empirical/statistical estimations from teff
    elif _is_valid_qty(teff):
        teff_val = teff.val
        logg_val = logg.val if _is_valid_qty(logg) else None
        met_val = met.val if _is_valid_qty(met) else None
        kmag_val = kmag.val if _is_valid_qty(kmag) else None
        dist_val = dist.val if _is_valid_qty(dist) else None

        r = 0.0
        err1 = err2 = 0
        r_tag = ""

        # 1. Mann 2015 M_Ks polynomial — best for K7–M7, about 3% scatter
        if kmag_val is not None and dist_val is not None and dist_val > 0 and teff_val < 4200:
            mks = kmag_val - 5.0 * np.log10(dist_val / 10.0)

            if 4.0 <= mks <= 10.5:
                r = _mann_mks_radius(mks, met_val)
                r_tag = f"mann_mks({_src('kmag', kmag)} {_src('dist', dist)})"
                err1 = err2 = 0.029 * r

        # 2. Torres 2010 — best for FGK with log g, about 3% scatter
        if r == 0.0 and logg_val is not None and 3900 < teff_val < 8500:
            r = _torres_radius(teff_val, logg_val, met_val)
            r_tag = f"torres({_src('teff', teff)} {_src('logg', logg)})"
            err1 = err2 = 0.030 * r

        # 3. Mann 2015 Teff polynomial — better than power law for M dwarfs
        if r == 0.0 and teff_val < 4000:
            r = _mann_teff_radius(teff_val, met_val)
            r_tag = f"mann_teff({_src('teff', teff)})"
            frac = 0.093 if (met_val is not None and np.isfinite(met_val)) else 0.134
            err1 = err2 = frac * r

        # 4. ZAMS power-law — last resort
        if r == 0.0:
            r = ms_radius_from_teff(teff_val, spectype.val)
            r_tag = f"ms({_src('teff', teff)})"
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
                err2=err2,
            )

    return rad


def infer_star_mass(
    mass: ParamQty,
    rad: ParamQty,
    logg: ParamQty,
) -> ParamQty:
    r"""Derive stellar mass if it is masked, using the physical log(g) relation.

    Calculates the mass from the stellar radius and surface gravity log(g):

    M = 10^(logg - logg_sun + 2 log10(R))

    Passes through unchanged if that is not possible.

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
    if _is_valid_qty(mass):
        return mass

    if _is_valid_qty(logg) and _is_valid_qty(rad) and rad.val > 0:
        # M_* = (g / g_sun) * R^2 = 10^(logg - logg_sun + 2 log10(R))
        m = 10 ** (logg.val - _LOG_G_SUN + 2 * np.log10(rad.val))

        # Error propagation
        # sigma_M = sqrt((M_* ln(10) sigma_logg)^2 + (2 M_* sigma_R / R)^2)
        t_r_up = _safe_err_sq(rad.err1, rad.val, 2.0)
        t_r_dn = _safe_err_sq(rad.err2, rad.val, 2.0)
        t_g_up = _safe_err_sq(logg.err1, coeff=np.log(10.0))
        t_g_dn = _safe_err_sq(logg.err2, coeff=np.log(10.0))

        err1 = m * np.sqrt(t_r_up + t_g_up)
        err2 = m * np.sqrt(t_r_dn + t_g_dn)

        return ParamQty(
            val=m,
            mask=False,
            src=f"logg_derived({_src('rad', rad)} {_src('logg', logg)})",
            err1=err1,
            err2=err2,
        )

    return mass


def infer_stellar_luminosity(
    lum: ParamQty,
    rad: ParamQty,
    teff: ParamQty,
) -> ParamQty:
    r"""Derive stellar luminosity if it is masked.

    Uses the Stefan-Boltzmann scaling relation in solar units:

    L = R^2 * (T / T_sun)^4

    Passes through the input luminosity unchanged if it already contains a
    valid value or if the required inputs are unavailable.

    Parameters
    ----------
    lum : ParamQty
        Stellar luminosity data in solar luminosities.
    rad : ParamQty
        Stellar radius data in solar radii.
    teff : ParamQty
        Stellar effective temperature data in Kelvin.

    Returns
    -------
    ParamQty
        A new ParamQty containing either the original luminosity or the derived luminosity.
    """
    if _is_valid_qty(lum):
        return lum

    if not _is_valid_qty(rad) or not _is_valid_qty(teff):
        return lum

    # Error propagation
    # sigma_L = L * sqrt(4 * (sigma_R / R)^2 + 16 * (sigma_T / T)^2)
    l_val = rad.val**2 * (teff.val / T_SUN) ** 4

    rad_up = _safe_err_sq(rad.err1, rad.val, 2.0)
    rad_dn = _safe_err_sq(rad.err2, rad.val, 2.0)
    teff_up = _safe_err_sq(teff.err1, teff.val, 4.0)
    teff_dn = _safe_err_sq(teff.err2, teff.val, 4.0)

    err1 = l_val * np.sqrt(rad_up + teff_up)
    err2 = l_val * np.sqrt(rad_dn + teff_dn)

    return ParamQty(
        val=l_val,
        mask=False,
        src=f"StephanBoltzmann_derived({_src('rad', rad)} {_src('teff', teff)})",
        err1=err1,
        err2=err2,
    )


def infer_msini_radius_bounds(
    planet_radius: ParamQty,
    msini: ParamQty,
    msini_sin_min: float = 0.3,
) -> tuple[ParamQty, ParamQty]:
    r"""Derive lower and upper radius bounds from minimum planet mass.

    If the planet radius is already provided, no bounds are inferred. Otherwise,
    this function converts the minimum mass from Jupiter masses to Earth masses
    and applies the Chen-Kipping mass-radius relation.

    The lower bound uses M_p sin(i) directly.
    The upper bound uses M_p sin(i) / sin(i_min) as a conservative estimate of the true mass.

    Parameters
    ----------
    planet_radius : ParamQty
        Planet radius data. If valid and positive, no bounds are produced.
    msini : ParamQty
        Minimum planet mass in Jupiter masses.
    msini_sin_min : float
        Minimum allowed value of sin inclination used for the upper bound.

    Returns
    -------
    tuple[ParamQty, ParamQty]
        Lower and upper radius-bound estimates in Earth radii.
    """
    lower = _masked_qty()
    upper = _masked_qty()

    radius_valid = _is_valid_qty(planet_radius) and planet_radius.val > 0
    msini_valid = _is_valid_qty(msini) and msini.val > 0

    if radius_valid or not msini_valid:
        return lower, upper

    msini_earth = msini.val * u.M_jup.to(u.M_earth)

    lower_val = mass_radius_chen_kipping(msini_earth)
    upper_val = mass_radius_chen_kipping(
        ((msini.val + msini.err1 if np.isfinite(msini.err1) else 0.0) * u.M_jup.to(u.M_earth)) / msini_sin_min
    )

    lower = ParamQty(
        val=lower_val,
        mask=False,
        src=f"direct({_src('msini', msini)})",
        err1=np.nan,
        err2=np.nan,
    )

    upper = ParamQty(
        val=upper_val,
        mask=False,
        src=f"derived({_src('posterior_i_estimated', msini)} sin_min:{msini_sin_min})",
        err1=np.nan,
        err2=np.nan,
    )

    return lower, upper


def infer_semi_major_axis(
    semi_major_axis: ParamQty,
    period: ParamQty,
    mass: ParamQty,
    period_src: str = "period",
) -> ParamQty:
    r"""Derive semi-major axis from orbital period and stellar mass.

    Uses Kepler's third law in astronomical units, solar masses, and days.

    Passes through the input semi-major axis if it is already valid and positive.

    Parameters
    ----------
    semi_major_axis : ParamQty
        Semi-major axis data in AU.
    period : ParamQty
        Orbital period data in days.
    mass : ParamQty
        Stellar mass data in solar masses.
    period_src : str, optional
        Source label to use for the period if the period itself has no useful
        source string.

    Returns
    -------
    ParamQty
        Semi-major axis data in AU.
    """
    if _is_valid_qty(semi_major_axis) and semi_major_axis.val > 0:
        return semi_major_axis

    if not (_is_valid_qty(period) and period.val > 0):
        return semi_major_axis

    if not (_is_valid_qty(mass) and mass.val > 0):
        return semi_major_axis

    a_val = (mass.val * (period.val / 365.25) ** 2) ** (1.0 / 3.0)

    err1 = a_val * np.sqrt(
        _safe_err_sq(mass.err1, mass.val, 1.0 / 3.0)
        + _safe_err_sq(period.err1, period.val, 2.0 / 3.0)
    )
    err2 = a_val * np.sqrt(
        _safe_err_sq(mass.err2, mass.val, 1.0 / 3.0)
        + _safe_err_sq(period.err2, period.val, 2.0 / 3.0)
    )

    return ParamQty(
        val=a_val,
        mask=False,
        src=f"kepler_3rd({_src('mass', mass)} {_src('period', period, period_src)})",
        err1=err1,
        err2=err2,
    )


def infer_planet_flux(
    insol: ParamQty,
    lum: ParamQty,
    eqt: ParamQty,
    semi_major_axis: ParamQty,
) -> ParamQty:
    r"""Derive planet insolation flux if it is masked.

    First attempts to derive insolation from luminosity and semi-major axis.

    If that is not possible, attempts to derive insolation from equilibrium
    temperature.

    Passes through the input insolation if it is already valid.

    Parameters
    ----------
    insol : ParamQty
        Planet insolation flux in Earth flux units.
    lum : ParamQty
        Stellar luminosity in solar luminosities.
    semi_major_axis : ParamQty
        Semi-major axis in AU.
    eqt : ParamQty
        Planet equilibrium temperature in Kelvin.

    Returns
    -------
    ParamQty
        Planet insolation flux in Earth flux units.
    """
    if _is_valid_qty(insol):
        return insol

    if (
        _is_valid_qty(lum)
        and _is_valid_qty(semi_major_axis)
        and semi_major_axis.val > 0
    ):
        # F = \frac{L_{\star}}{a^2}    
        s_val = lum.val / semi_major_axis.val**2

        # \sigma_F = F_E \sqrt{\left(\frac{\sigma_L}{L_{\star}}\right)^2 + 4\left(\frac{\sigma_a}{a}\right)^2}
        lum_up = _safe_err_sq(lum.err1, lum.val, 1.0)
        lum_dn = _safe_err_sq(lum.err2, lum.val, 1.0)

        a_dn = _safe_err_sq(semi_major_axis.err2, semi_major_axis.val, 2.0)
        a_up = _safe_err_sq(semi_major_axis.err1, semi_major_axis.val, 2.0)

        err1 = s_val * np.sqrt(lum_up + a_dn)
        err2 = s_val * np.sqrt(lum_dn + a_up)
        src = f"derived({_src('lum', lum)} {_src('a', semi_major_axis)})"

        return ParamQty(
            val=s_val,
            mask=False,
            src=src,
            err1=err1,
            err2=err2,
        )

    if _is_valid_qty(eqt) and eqt.val > 0:
        # F = \left(\frac{T_{eq}}{255}\right)^4
        s_val = (eqt.val / 254.793) ** 4

        err1 = abs(s_val) * np.sqrt(_safe_err_sq(eqt.err1, eqt.val, 4.0))
        err2 = abs(s_val) * np.sqrt(_safe_err_sq(eqt.err2, eqt.val, 4.0))

        return ParamQty(
            val=s_val,
            mask=False,
            src=f"derived({_src('eqt', eqt)})",
            err1=err1,
            err2=err2,
        )

    return insol


def infer_planet_equilibrium_temperature(
    eqt: ParamQty,
    insol: ParamQty,
    albedo: float = 0.3
) -> ParamQty:
    r"""Derive planet equilibrium temperature if it is masked.

    Uses the insolation-temperature scaling, assuming an albedo of 0.3. 

    $$ T_{\t}

    Passes through the input equilibrium temperature if it is already valid.

    Parameters
    ----------
    eqt : ParamQty
        Planet equilibrium temperature in Kelvin.
    insol : ParamQty
        Planet insolation flux in Earth flux units.

    Returns
    -------
    ParamQty
        Planet equilibrium temperature in Kelvin.
    """
    if _is_valid_qty(eqt):
        return eqt

    if not _is_valid_qty(insol):
        return eqt

    albedo_factor= (1-albedo)/(1-0.3) # 0.3 is earth's albedo
    t_val = 255 * (insol.val*albedo_factor)**0.25

    # \frac{\sigma_T}{T_{eq}} = \frac{1}{4} \frac{\sigma_F}{F_E}
    err1 = t_val * np.sqrt(_safe_err_sq(insol.err1, insol.val, 0.25))
    err2 = t_val * np.sqrt(_safe_err_sq(insol.err2, insol.val, 0.25))

    return ParamQty(
        val=t_val,
        mask=False,
        src=f"derived({_src('insol', insol)})",
        err1=err1,
        err2=err2,
    )


def infer_spectral_type_display(
    spectype: ParamStr,
    teff: ParamQty,
) -> ParamStr:
    """Return a display-ready stellar spectral type.

    Uses the available spectral type string and, if available, the stellar
    effective temperature to construct the display value.

    Parameters
    ----------
    spectype : ParamStr
        Stellar spectral type data.
    teff : ParamQty
        Stellar effective temperature data.

    Returns
    -------
    ParamStr
        Display-ready spectral type string.
    """
    teff_val = teff.val if _is_valid_qty(teff) else 0.0

    displayed = spectype_display(
        spectype.val if not spectype.mask else "",
        teff_val,
    )

    return ParamStr(
        val=displayed,
        mask=not bool(str(displayed).strip()),
        src=spectype.src,
    )