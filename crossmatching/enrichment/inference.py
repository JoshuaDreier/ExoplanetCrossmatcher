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
from crossmatching.enrichment.spectral_types import spectype_to_teff, get_spectral_class_range, spectype_display


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


_M_JUP_TO_EARTH = u.M_jup.to(u.M_earth)

def _is_valid_qty(q: ParamQty) -> bool:
    """Return True if a ParamQty contains a usable finite value."""
    return not q.mask and np.isfinite(q.val)


def _copy_qty(q: ParamQty) -> ParamQty:
    """Return a shallow value-copy of a ParamQty."""
    return ParamQty(
        val=q.val,
        mask=q.mask,
        src=q.src,
        err1=q.err1,
        err2=q.err2,
    )


def _copy_str(s: ParamStr) -> ParamStr:
    """Return a shallow value-copy of a ParamStr."""
    return ParamStr(
        val=s.val,
        mask=s.mask,
        src=s.src,
    )


def _masked_qty(src: str = "") -> ParamQty:
    """Return a masked ParamQty with NaN value and uncertainties."""
    return ParamQty(
        val=np.nan,
        mask=True,
        src=src,
        err1=np.nan,
        err2=np.nan,
    )



def infer_stellar_luminosity(
    lum: ParamQty,
    rad: ParamQty,
    teff: ParamQty,
) -> ParamQty:
    r"""Derive stellar luminosity if it is masked.

    Uses the Stefan-Boltzmann scaling relation in solar units:

    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" style="font-size: 121.4%; position: relative;" display="true" tabindex="0" ctxtmenu_counter="8"><mjx-math display="true" style="margin-left: 0px; margin-right: 0px;" class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43F TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-msup space="4"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D445 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: 0.413em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-script></mjx-msup><mjx-msup><mjx-mrow><mjx-mo class="mjx-s3"><mjx-c class="mjx-c28 TEX-S3"></mjx-c></mjx-mo><mjx-mfrac><mjx-frac type="d"><mjx-num><mjx-nstrut type="d"></mjx-nstrut><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.12em;"><mjx-texatom size="s" texclass="ORD"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c65"></mjx-c><mjx-c class="mjx-c66"></mjx-c><mjx-c class="mjx-c66"></mjx-c></mjx-mi></mjx-texatom></mjx-texatom></mjx-script></mjx-msub></mjx-num><mjx-dbox><mjx-dtable><mjx-line type="d"></mjx-line><mjx-row><mjx-den><mjx-dstrut type="d"></mjx-dstrut><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.12em;"><mjx-mo class="mjx-n" size="s"><mjx-c class="mjx-c2299"></mjx-c></mjx-mo></mjx-script></mjx-msub></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-s3"><mjx-c class="mjx-c29 TEX-S3"></mjx-c></mjx-mo></mjx-mrow><mjx-script style="vertical-align: 1.177em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c34"></mjx-c></mjx-mn></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>L</mi><mo>=</mo><msup><mi>R</mi><mn>2</mn></msup><msup><mrow data-mjx-texclass="INNER"><mo data-mjx-texclass="OPEN">(</mo><mfrac><msub><mi>T</mi><mrow data-mjx-texclass="ORD"><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">eff</mi></mrow></mrow></msub><msub><mi>T</mi><mo>⊙</mo></msub></mfrac><mo data-mjx-texclass="CLOSE">)</mo></mrow><mn>4</mn></msup></math></mjx-assistive-mml></mjx-container>

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
        A new ParamQty containing either the original luminosity or the derived
        luminosity.
    """
    if _is_valid_qty(lum):
        return _copy_qty(lum)

    if not _is_valid_qty(rad) or not _is_valid_qty(teff):
        return _copy_qty(lum)

    l_val = rad.val**2 * (teff.val / T_SUN) ** 4

    rad_up = _safe_err_sq(rad.err1, rad.val, 2.0)
    rad_dn = _safe_err_sq(rad.err2, rad.val, 2.0)
    teff_up = _safe_err_sq(teff.err1, teff.val, 4.0)
    teff_dn = _safe_err_sq(teff.err2, teff.val, 4.0)

    err1_sq = rad_up + teff_up
    err2_sq = rad_dn + teff_dn

    err1 = abs(l_val) * np.sqrt(err1_sq) if err1_sq > 0 else np.nan
    err2 = abs(l_val) * np.sqrt(err2_sq) if err2_sq > 0 else np.nan

    return ParamQty(
        val=l_val,
        mask=False,
        src=f"derived(rad:{rad.src} teff:{teff.src})",
        err1=err1,
        err2=err2,
    )


def infer_msini_radius_bounds(
    planet_radius: ParamQty,
    msini: ParamQty,
    msini_sin_min: float,
) -> tuple[ParamQty, ParamQty]:
    r"""Derive lower and upper radius bounds from minimum planet mass.

    If the planet radius is already provided, no bounds are inferred. Otherwise,
    this function converts the minimum mass from Jupiter masses to Earth masses
    and applies the Chen-Kipping mass-radius relation.

    The lower bound uses

    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" style="font-size: 121.4%; position: relative;" display="true" tabindex="0" ctxtmenu_counter="9"><mjx-math display="true" style="margin-left: 0px; margin-right: 0px;" class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c70"></mjx-c></mjx-mi></mjx-texatom></mjx-texatom></mjx-script></mjx-msub><mjx-mi class="mjx-n" space="2"><mjx-c class="mjx-c73"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6E"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2061"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D456 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false" mathvariant="normal">p</mi></mrow></mrow></msub><mi>sin</mi><mo data-mjx-texclass="NONE">⁡</mo><mi>i</mi></math></mjx-assistive-mml></mjx-container>

    directly. The upper bound uses

    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" style="font-size: 121.4%; position: relative;" display="true" tabindex="0" ctxtmenu_counter="10"><mjx-math display="true" style="margin-left: 0px; margin-right: 0px;" class="MJX-TEX" aria-hidden="true"><mjx-mfrac><mjx-frac type="d"><mjx-num><mjx-nstrut type="d"></mjx-nstrut><mjx-mrow><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c70"></mjx-c></mjx-mi></mjx-texatom></mjx-texatom></mjx-script></mjx-msub><mjx-mi class="mjx-n" space="2"><mjx-c class="mjx-c73"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6E"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2061"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D456 TEX-I"></mjx-c></mjx-mi></mjx-mrow></mjx-num><mjx-dbox><mjx-dtable><mjx-line type="d"></mjx-line><mjx-row><mjx-den><mjx-dstrut type="d"></mjx-dstrut><mjx-mrow><mjx-mi class="mjx-n"><mjx-c class="mjx-c73"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6E"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2061"></mjx-c></mjx-mo><mjx-msub space="2"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D456 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-texatom size="s" texclass="ORD"><mjx-mo class="mjx-n"><mjx-c class="mjx-c6D"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6E"></mjx-c></mjx-mo></mjx-texatom></mjx-script></mjx-msub></mjx-mrow></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mfrac><mrow><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false" mathvariant="normal">p</mi></mrow></mrow></msub><mi>sin</mi><mo data-mjx-texclass="NONE">⁡</mo><mi>i</mi></mrow><mrow><mi>sin</mi><mo data-mjx-texclass="NONE">⁡</mo><msub><mi>i</mi><mrow data-mjx-texclass="ORD"><mo data-mjx-texclass="OP" movablelimits="true">min</mo></mrow></msub></mrow></mfrac></math></mjx-assistive-mml></mjx-container>

    as a conservative estimate of the true mass.

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

    msini_earth = msini.val * _M_JUP_TO_EARTH

    lower_val = mass_radius_chen_kipping(msini_earth)
    upper_val = mass_radius_chen_kipping(msini_earth / msini_sin_min)

    lower = ParamQty(
        val=lower_val,
        mask=False,
        src=f"derived(msini:{msini.src})",
        err1=np.nan,
        err2=np.nan,
    )

    upper = ParamQty(
        val=upper_val,
        mask=False,
        src=f"derived(msini:{msini.src} sin_min:{msini_sin_min})",
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

    Uses Kepler's third law in astronomical units, solar masses, and days:

    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" style="font-size: 121.4%; position: relative;" display="true" tabindex="0" ctxtmenu_counter="11"><mjx-math display="true" style="margin-left: 0px; margin-right: 0px;" class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44E TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-msup space="4"><mjx-mrow><mjx-mo class="mjx-s4"><mjx-c class="mjx-c28 TEX-S4"></mjx-c></mjx-mo><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-mo class="mjx-n" size="s"><mjx-c class="mjx-c22C6"></mjx-c></mjx-mo></mjx-script></mjx-msub><mjx-msup><mjx-mrow><mjx-mo class="mjx-s3"><mjx-c class="mjx-c28 TEX-S3"></mjx-c></mjx-mo><mjx-mfrac><mjx-frac type="d"><mjx-num><mjx-nstrut type="d"></mjx-nstrut><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D443 TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line type="d"></mjx-line><mjx-row><mjx-den><mjx-dstrut type="d"></mjx-dstrut><mjx-mn class="mjx-n"><mjx-c class="mjx-c33"></mjx-c><mjx-c class="mjx-c36"></mjx-c><mjx-c class="mjx-c35"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c35"></mjx-c></mjx-mn></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-s3"><mjx-c class="mjx-c29 TEX-S3"></mjx-c></mjx-mo></mjx-mrow><mjx-script style="vertical-align: 1.177em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-script></mjx-msup><mjx-mo class="mjx-s4"><mjx-c class="mjx-c29 TEX-S4"></mjx-c></mjx-mo></mjx-mrow><mjx-script style="vertical-align: 1.477em;"><mjx-texatom size="s" texclass="ORD"><mjx-mn class="mjx-n"><mjx-c class="mjx-c31"></mjx-c></mjx-mn><mjx-texatom texclass="ORD"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2F"></mjx-c></mjx-mo></mjx-texatom><mjx-mn class="mjx-n"><mjx-c class="mjx-c33"></mjx-c></mjx-mn></mjx-texatom></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>a</mi><mo>=</mo><msup><mrow data-mjx-texclass="INNER"><mo data-mjx-texclass="OPEN">(</mo><msub><mi>M</mi><mo>⋆</mo></msub><msup><mrow data-mjx-texclass="INNER"><mo data-mjx-texclass="OPEN">(</mo><mfrac><mi>P</mi><mn>365.25</mn></mfrac><mo data-mjx-texclass="CLOSE">)</mo></mrow><mn>2</mn></msup><mo data-mjx-texclass="CLOSE">)</mo></mrow><mrow data-mjx-texclass="ORD"><mn>1</mn><mrow data-mjx-texclass="ORD"><mo>/</mo></mrow><mn>3</mn></mrow></msup></math></mjx-assistive-mml></mjx-container>

    Passes through the input semi-major axis if it is already valid and
    positive.

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
        return _copy_qty(semi_major_axis)

    if not (_is_valid_qty(period) and period.val > 0):
        return _copy_qty(semi_major_axis)

    if not (_is_valid_qty(mass) and mass.val > 0):
        return _copy_qty(semi_major_axis)

    a_val = (mass.val * (period.val / 365.25) ** 2) ** (1.0 / 3.0)

    err1 = np.nan
    err2 = np.nan

    if np.isfinite(mass.err1):
        err1 = (1.0 / 3.0) * a_val * mass.err1 / mass.val

    if np.isfinite(mass.err2):
        err2 = (1.0 / 3.0) * a_val * mass.err2 / mass.val

    src_period = period.src or period_src

    return ParamQty(
        val=a_val,
        mask=False,
        src=f"kepler(mass:{mass.src} period:{src_period})",
        err1=err1,
        err2=err2,
    )


def infer_planet_insolation(
    insol: ParamQty,
    lum: ParamQty,
    semi_major_axis: ParamQty,
    eqt: ParamQty,
) -> ParamQty:
    r"""Derive planet insolation flux if it is masked.

    First attempts to derive insolation from luminosity and semi-major axis:

    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" style="font-size: 121.4%; position: relative;" display="true" tabindex="0" ctxtmenu_counter="12"><mjx-math display="true" style="margin-left: 0px; margin-right: 0px;" class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mfrac space="4"><mjx-frac type="d"><mjx-num><mjx-nstrut type="d"></mjx-nstrut><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43F TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line type="d"></mjx-line><mjx-row><mjx-den><mjx-dstrut type="d"></mjx-dstrut><mjx-msup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: 0.289em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-script></mjx-msup></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>S</mi><mo>=</mo><mfrac><mi>L</mi><msup><mi>a</mi><mn>2</mn></msup></mfrac></math></mjx-assistive-mml></mjx-container>

    If that is not possible, attempts to derive insolation from equilibrium
    temperature:

    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" style="font-size: 121.4%; position: relative;" display="true" tabindex="0" ctxtmenu_counter="13"><mjx-math display="true" style="margin-left: 0px; margin-right: 0px;" class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-msup space="4"><mjx-mrow><mjx-mo class="mjx-s3"><mjx-c class="mjx-c28 TEX-S3"></mjx-c></mjx-mo><mjx-mfrac><mjx-frac type="d"><mjx-num><mjx-nstrut type="d"></mjx-nstrut><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.12em;"><mjx-texatom size="s" texclass="ORD"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c65"></mjx-c><mjx-c class="mjx-c71"></mjx-c></mjx-mi></mjx-texatom></mjx-texatom></mjx-script></mjx-msub></mjx-num><mjx-dbox><mjx-dtable><mjx-line type="d"></mjx-line><mjx-row><mjx-den><mjx-dstrut type="d"></mjx-dstrut><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c35"></mjx-c><mjx-c class="mjx-c34"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c37"></mjx-c><mjx-c class="mjx-c39"></mjx-c><mjx-c class="mjx-c33"></mjx-c></mjx-mn></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-s3"><mjx-c class="mjx-c29 TEX-S3"></mjx-c></mjx-mo></mjx-mrow><mjx-script style="vertical-align: 1.177em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c34"></mjx-c></mjx-mn></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>S</mi><mo>=</mo><msup><mrow data-mjx-texclass="INNER"><mo data-mjx-texclass="OPEN">(</mo><mfrac><msub><mi>T</mi><mrow data-mjx-texclass="ORD"><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">eq</mi></mrow></mrow></msub><mn>254.793</mn></mfrac><mo data-mjx-texclass="CLOSE">)</mo></mrow><mn>4</mn></msup></math></mjx-assistive-mml></mjx-container>

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
        return _copy_qty(insol)

    if (
        _is_valid_qty(lum)
        and _is_valid_qty(semi_major_axis)
        and semi_major_axis.val > 0
    ):
        s_val = lum.val / semi_major_axis.val**2

        lum_up = _safe_err_sq(lum.err1, lum.val, 1.0)
        lum_dn = _safe_err_sq(lum.err2, lum.val, 1.0)

        a_dn = _safe_err_sq(semi_major_axis.err2, semi_major_axis.val, 2.0)
        a_up = _safe_err_sq(semi_major_axis.err1, semi_major_axis.val, 2.0)

        err1_sq = lum_up + a_dn
        err2_sq = lum_dn + a_up

        err1 = abs(s_val) * np.sqrt(err1_sq) if err1_sq > 0 else np.nan
        err2 = abs(s_val) * np.sqrt(err2_sq) if err2_sq > 0 else np.nan
        lum_src = lum.src
        if lum_src.startswith("derived(rad:") and " teff:" in lum_src and lum_src.endswith(")"):
            inner = lum_src[len("derived(rad:"):-1]
            rad_src, teff_src = inner.split(" teff:", 1)
            src = f"derived(r:{rad_src} teff:{teff_src} a:{semi_major_axis.src})"
        else:
            src = f"derived(lum:{lum_src} a:{semi_major_axis.src})"

        return ParamQty(
            val=s_val,
            mask=False,
            src=src,
            err1=err1,
            err2=err2,
        )

    if _is_valid_qty(eqt) and eqt.val > 0:
        s_val = (eqt.val / 254.793) ** 4

        err1 = np.nan
        err2 = np.nan

        if np.isfinite(eqt.err1):
            err1 = 4.0 * s_val * eqt.err1 / eqt.val

        if np.isfinite(eqt.err2):
            err2 = 4.0 * s_val * eqt.err2 / eqt.val

        return ParamQty(
            val=s_val,
            mask=False,
            src=f"derived(eqt:{eqt.src})",
            err1=err1,
            err2=err2,
        )

    return _copy_qty(insol)


def infer_planet_equilibrium_temperature(
    eqt: ParamQty,
    insol: ParamQty,
) -> ParamQty:
    r"""Derive planet equilibrium temperature if it is masked.

    Uses the insolation-temperature scaling

    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" style="font-size: 121.4%; position: relative;" display="true" tabindex="0" ctxtmenu_counter="14"><mjx-math display="true" style="margin-left: 0px; margin-right: 0px;" class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.12em;"><mjx-texatom size="s" texclass="ORD"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c65"></mjx-c><mjx-c class="mjx-c71"></mjx-c></mjx-mi></mjx-texatom></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c35"></mjx-c><mjx-c class="mjx-c34"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c37"></mjx-c><mjx-c class="mjx-c39"></mjx-c><mjx-c class="mjx-c33"></mjx-c></mjx-mn><mjx-msup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: 0.413em; margin-left: 0.052em;"><mjx-texatom size="s" texclass="ORD"><mjx-mn class="mjx-n"><mjx-c class="mjx-c31"></mjx-c></mjx-mn><mjx-texatom texclass="ORD"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2F"></mjx-c></mjx-mo></mjx-texatom><mjx-mn class="mjx-n"><mjx-c class="mjx-c34"></mjx-c></mjx-mn></mjx-texatom></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>T</mi><mrow data-mjx-texclass="ORD"><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">eq</mi></mrow></mrow></msub><mo>=</mo><mn>254.793</mn><msup><mi>S</mi><mrow data-mjx-texclass="ORD"><mn>1</mn><mrow data-mjx-texclass="ORD"><mo>/</mo></mrow><mn>4</mn></mrow></msup></math></mjx-assistive-mml></mjx-container>

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
        return _copy_qty(eqt)

    if not (_is_valid_qty(insol) and insol.val > 0):
        return _copy_qty(eqt)

    t_val = 254.793 * insol.val**0.25

    err1 = np.nan
    err2 = np.nan

    if np.isfinite(insol.err1):
        err1 = 0.25 * t_val * insol.err1 / insol.val

    if np.isfinite(insol.err2):
        err2 = 0.25 * t_val * insol.err2 / insol.val

    return ParamQty(
        val=t_val,
        mask=False,
        src=f"derived(insol:{insol.src})",
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
