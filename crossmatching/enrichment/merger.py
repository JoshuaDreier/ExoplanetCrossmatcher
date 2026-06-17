from __future__ import annotations
import numpy as np
from astropy import units as u
from astropy.table import MaskedColumn, Table

from crossmatching import config
from crossmatching import IdSupplierBase
from crossmatching.enrichment.param_sources.base import ParamSource
from crossmatching.enrichment.spectral_types import (
    classify_spectral_type,
    spectype_display,
)
from crossmatching.enrichment.radius_estimation import (
    _mann_mks_radius,
    _mann_teff_radius,
    _torres_radius,
    _zams_exponent,
    ms_radius_from_teff,
    mass_radius_chen_kipping,
    T_SUN,
)

_LOG_G_SUN = 4.438    # log10(g_sun / cm s-2), IAU 2015 nominal solar values
_M_JUP_TO_EARTH = u.M_jup.to(u.M_earth)  # ~317.83
R_JUP_TO_EARTH  = u.R_jup.to(u.R_earth)  # ~11.21; exported for callers that need r in R_earth


def _col_float(table: Table, col: str):
    """Return (values, mask) float arrays for a table column.

    Returns an all-masked pair when the column is absent, so the
    optional EMC-schema planet columns (``r``, ``a``, ``p``, ``msini``)
    degrade gracefully on tables with other layouts.

    Parameters
    ----------
    table : Table
        The input astropy table.
    col : str
        The name of the column to extract.

    Returns
    -------
    values : np.ndarray
        Float array of the column's values.
    mask : np.ndarray
        Boolean array indicating masked (missing) values.
    """
    if col not in table.colnames:
        n = len(table)
        return np.zeros(n), np.ones(n, bool)
    
    c = table[col]
    if hasattr(c, 'mask'):
        mask = np.array(c.mask, dtype=bool)
        if mask.ndim == 0:
            mask = np.full(len(c), mask.item(), dtype=bool)
    else:
        mask = np.zeros(len(c), bool)
    return np.array(c, dtype=float), mask

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


def _derive_stellar_teff(
    i: int,
    teff: _ParamQtyArrays,
    rad: _ParamQtyArrays,
    lum: _ParamQtyArrays,
    spec_arr: _ParamStrArrays,
) -> None:
    r"""Derive stellar effective temperature if it is masked, using physical or statistical relations.

    Calculates the temperature using the Stefan-Boltzmann law if radius and luminosity
    are available:
    
    $$ T_{\\text{eff}} = T_{\\odot} \\left( \\frac{L}{R^2} \\right)^{1/4} $$

    If not, estimates the temperature from the spectral type.

    Parameters
    ----------
    i : int
        The row index currently being processed.
    teff : _ParamQtyArrays
        Arrays containing stellar effective temperature data (modified in-place).
    rad : _ParamQtyArrays
        Arrays containing stellar radius data.
    lum : _ParamQtyArrays
        Arrays containing stellar luminosity data.
    spec_arr : _ParamStrArrays
        Arrays containing stellar spectral type data.
    """
    if not teff.mask[i]:
        return

    # 1. Try exact physical Stefan-Boltzmann relation first
    if not rad.mask[i] and not lum.mask[i] and rad.val[i] > 0 and lum.val[i] > 0:
        t = T_SUN * (lum.val[i] / (rad.val[i]**2))**0.25
        teff.val[i] = t
        teff.mask[i] = False
        teff.src[i] = f"SB_derived(rad:{rad.src[i]} lum:{lum.src[i]})"
        
        # Error propagation
        t_l_up = _safe_err_sq(lum.err1[i], lum.val[i], 0.25)
        t_l_dn = _safe_err_sq(lum.err2[i], lum.val[i], 0.25)
        t_r_up = _safe_err_sq(rad.err1[i], rad.val[i], 0.5)
        t_r_dn = _safe_err_sq(rad.err2[i], rad.val[i], 0.5)
        
        teff.err1[i] = t * np.sqrt(t_l_up + t_r_dn) if (t_l_up > 0 or t_r_dn > 0) else 0.0
        teff.err2[i] = t * np.sqrt(t_l_dn + t_r_up) if (t_l_dn > 0 or t_r_up > 0) else 0.0
        
    # 2. Try estimation from spectral type
    elif not spec_arr.mask[i]:
        spec_str = spec_arr.val[i]
        try:
            from crossmatching.enrichment.spectral_types import spectype_to_teff, get_spectral_class_range
            t = spectype_to_teff(spec_str)
            if t > 0:
                teff.val[i] = t
                teff.mask[i] = False
                teff.src[i] = f"spectype_derived(spec:{spec_arr.src[i]})"
                
                t_min, t_max = get_spectral_class_range(spec_str)
                if t_min > 0 and t_max > 0:
                    teff.err1[i] = t_max - t
                    teff.err2[i] = t - t_min
                else:
                    teff.err1[i] = teff.err2[i] = 0.0
        except ValueError:
            pass


def _derive_stellar_radius(
    i: int,
    rad: _ParamQtyArrays,
    mass: _ParamQtyArrays,
    logg: _ParamQtyArrays,
    teff: _ParamQtyArrays,
    lum: _ParamQtyArrays,
    met: _ParamQtyArrays,
    kmag: _ParamQtyArrays,
    dist: _ParamQtyArrays,
    spec_val: str,
) -> None:
    r"""Derive stellar radius if it is masked, using physical or statistical relations.

    Attempts to derive the radius in order of preference:
    1. Exact physical relation from mass and log(g):
       
       $$ R = 10^{0.5(4.43797 + \\log_{10} M - \\log g)} $$
       
    2. Exact physical relation from luminosity and Teff (Stefan-Boltzmann):
       
       $$ R = \\sqrt{L} \\left( \\frac{T_{\\odot}}{T_{\\text{eff}}} \\right)^2 $$
       
    3. Empirical/statistical polynomial relations based on Teff and other parameters

    Parameters
    ----------
    i : int
        The row index currently being processed.
    rad : _ParamQtyArrays
        Arrays containing stellar radius data (modified in-place).
    mass : _ParamQtyArrays
        Arrays containing stellar mass data.
    logg : _ParamQtyArrays
        Arrays containing surface gravity data.
    teff : _ParamQtyArrays
        Arrays containing stellar effective temperature data.
    lum : _ParamQtyArrays
        Arrays containing stellar luminosity data.
    met : _ParamQtyArrays
        Arrays containing stellar metallicity data.
    kmag : _ParamQtyArrays
        Arrays containing 2MASS K-band magnitude data.
    dist : _ParamQtyArrays
        Arrays containing distance data.
    spec_val : str
        The spectral type string for the star.
    """
    if not rad.mask[i]:
        return

    # 1. Try exact physical logg relation first
    if not logg.mask[i] and not mass.mask[i]:
        r = 10 ** (0.5 * (4.43797 + np.log10(mass.val[i]) - logg.val[i]))
        rad.val[i] = r
        rad.mask[i] = False
        rad.src[i] = f"logg_derived(mass:{mass.src[i]} logg:{logg.src[i]})"
        
        # Error propagation
        t_m_up = _safe_err_sq(mass.err1[i], mass.val[i], 0.5)
        t_m_dn = _safe_err_sq(mass.err2[i], mass.val[i], 0.5)
        t_g_up = _safe_err_sq(logg.err1[i], coeff=0.5 * np.log(10.0))
        t_g_dn = _safe_err_sq(logg.err2[i], coeff=0.5 * np.log(10.0))
        
        rad.err1[i] = r * np.sqrt(t_m_up + t_g_dn) if (t_m_up > 0 or t_g_dn > 0) else 0.0
        rad.err2[i] = r * np.sqrt(t_m_dn + t_g_up) if (t_m_dn > 0 or t_g_up > 0) else 0.0
        
    # 1.5. Try exact physical Stefan-Boltzmann relation next
    elif not lum.mask[i] and not teff.mask[i] and lum.val[i] > 0 and teff.val[i] > 0:
        r = np.sqrt(lum.val[i]) * (T_SUN / teff.val[i]) ** 2
        rad.val[i] = r
        rad.mask[i] = False
        rad.src[i] = f"SB_derived(lum:{lum.src[i]} teff:{teff.src[i]})"
        
        # Error propagation
        t_l_up = _safe_err_sq(lum.err1[i], lum.val[i], 0.5)
        t_l_dn = _safe_err_sq(lum.err2[i], lum.val[i], 0.5)
        t_t_up = _safe_err_sq(teff.err1[i], teff.val[i], 2.0)
        t_t_dn = _safe_err_sq(teff.err2[i], teff.val[i], 2.0)
        
        rad.err1[i] = r * np.sqrt(t_l_up + t_t_dn) if (t_l_up > 0 or t_t_dn > 0) else 0.0
        rad.err2[i] = r * np.sqrt(t_l_dn + t_t_up) if (t_l_dn > 0 or t_t_up > 0) else 0.0

    # 2. Try empirical/statistical estimations from teff
    elif not teff.mask[i]:
        teff_val = teff.val[i]
        logg_val = logg.val[i] if not logg.mask[i] else None
        met_val  = met.val[i] if not met.mask[i] else None
        kmag_val = kmag.val[i] if not kmag.mask[i] else None
        dist_val = dist.val[i] if not dist.mask[i] else None

        r = 0.0
        r_tag = ''

        # 1. Mann 2015 M_Ks polynomial — best for K7–M7 (~3% scatter)
        if kmag_val is not None and dist_val is not None and dist_val > 0 and teff_val < 4200:
            mks = kmag_val - 5.0 * np.log10(dist_val / 10.0)
            if 4.0 <= mks <= 10.5:
                r = _mann_mks_radius(mks, met_val)
                r_tag = f"mann_mks(kmag:{kmag.src[i]})"

        # 2. Torres 2010 — best for FGK with log g (~3% scatter)
        if r == 0.0 and logg_val is not None and 3900 < teff_val < 8500:
            r = _torres_radius(teff_val, logg_val, met_val)
            r_tag = f"torres(teff:{teff.src[i]} logg:{logg.src[i]})"

        # 3. Mann 2015 Teff polynomial — better than power law for M dwarfs
        if r == 0.0 and teff_val < 4000:
            r = _mann_teff_radius(teff_val, met_val)
            r_tag = f"mann_teff(teff:{teff.src[i]})"

        # 4. ZAMS power‑law — last resort
        if r == 0.0:
            r = ms_radius_from_teff(teff_val, spec_val)
            r_tag = f"ms(teff:{teff.src[i]})"

        if r > 0:
            rad.val[i]  = r
            rad.mask[i] = False
            rad.src[i]  = r_tag
            if r_tag.startswith('mann_mks'):
                rad.err1[i] = rad.err2[i] = 0.029 * r
            elif r_tag.startswith('torres'):
                rad.err1[i] = rad.err2[i] = 0.030 * r
            elif r_tag.startswith('mann_teff'):
                frac = 0.093 if (met_val is not None and np.isfinite(met_val)) else 0.134
                rad.err1[i] = rad.err2[i] = frac * r
            elif r > 0.05 and teff_val > 0:
                exp = _zams_exponent(teff_val, spec_val)
                coeff = exp * r / teff_val
                if np.isfinite(teff.err1[i]):
                    rad.err1[i] = coeff * teff.err1[i]
                if np.isfinite(teff.err2[i]):
                    rad.err2[i] = coeff * teff.err2[i]
            else:
                rad.err1[i] = rad.err2[i] = 0.0


def _derive_stellar_mass(
    i: int,
    mass: _ParamQtyArrays,
    rad: _ParamQtyArrays,
    logg: _ParamQtyArrays,
) -> None:
    r"""Derive stellar mass if it is masked, using the physical log(g) relation.

    Calculates the mass from the stellar radius and surface gravity log(g):
    
    $$ M = 10^{\\log g - 4.43797 + 2 \\log_{10} R} $$

    Parameters
    ----------
    i : int
        The row index currently being processed.
    mass : _ParamQtyArrays
        Arrays containing stellar mass data (modified in-place).
    rad : _ParamQtyArrays
        Arrays containing stellar radius data.
    logg : _ParamQtyArrays
        Arrays containing surface gravity data.
    """
    if not mass.mask[i]:
        return

    if not logg.mask[i] and not rad.mask[i]:
        m = 10 ** (logg.val[i] - 4.43797 + 2 * np.log10(rad.val[i]))
        mass.val[i] = m
        mass.mask[i] = False
        mass.src[i] = f"logg_derived(rad:{rad.src[i]} logg:{logg.src[i]})"
        
        # Error propagation
        t_r_up = _safe_err_sq(rad.err1[i], rad.val[i], 2.0)
        t_r_dn = _safe_err_sq(rad.err2[i], rad.val[i], 2.0)
        t_g_up = _safe_err_sq(logg.err1[i], coeff=np.log(10.0))
        t_g_dn = _safe_err_sq(logg.err2[i], coeff=np.log(10.0))
        
        mass.err1[i] = m * np.sqrt(t_r_up + t_g_up) if (t_r_up > 0 or t_g_up > 0) else 0.0
        mass.err2[i] = m * np.sqrt(t_r_dn + t_g_dn) if (t_r_dn > 0 or t_g_dn > 0) else 0.0


class _ParamQtyArrays:
    """Arrays for one merged parameter: value, mask, provenance, and errors.

    This class serves as a struct to hold the combined data for a single physical quantity
    across all rows of a table during the enrichment process.

    Attributes
    ----------
    val : np.ndarray
        Array of the primary values for the parameter.
    mask : np.ndarray
        Boolean array where True indicates the value is missing or masked.
    src : list of str
        Provenance string for each row indicating where the value came from.
    err1 : np.ndarray
        Upper 1σ uncertainty as a positive magnitude; NaN if absent.
    err2 : np.ndarray
        Lower 1σ uncertainty as a positive magnitude; NaN if absent.
    """

    def __init__(self, n: int, fill: float = 0.0):
        self.val  = np.full(n, fill)
        self.mask = np.ones(n, bool)
        self.src  = [''] * n
        self.err1 = np.full(n, np.nan)
        self.err2 = np.full(n, np.nan)


class _ParamStrArrays:
    """Arrays for one merged string parameter: value, mask, and provenance.

    This class serves as a struct to hold the combined string data (e.g. spectral type)
    across all rows of a table during the enrichment process.

    Attributes
    ----------
    val : list of str
        List of the primary string values for the parameter.
    mask : np.ndarray
        Boolean array where True indicates the value is missing, empty, or None.
    src : list of str
        Provenance string for each row indicating where the value came from.
    """
    def __init__(self, n, fill: str = ''):
        self.val = [fill] * n
        self.mask = np.ones(n, bool)
        self.src = [''] * n


class ParamFiller:
    """Enriches a catalog table with stellar and derived planetary parameters.

    Applies a priority-ordered chain of ParamSource objects: for each
    row the first source that provides a value for a given parameter wins.
    Computes derived physical columns (r_lower_bound, r_upper_bound, a, pl_insol, spectral_category)
    from the merged parameters.
    """

    def __init__(self, sources: list[ParamSource], msini_sin_min: float = 0.3):
        """Initialise the merger with an ordered list of parameter sources.

        Parameters
        ----------
        sources : list of `~crossmatching.param_sources.base.ParamSource`
            Priority-ordered parameter sources.  For each  parameter the 
            first source that provides a non-null value wins.
        msini_sin_min : float, optional
            Minimum sin(inclination) assumed when converting msini to a
            radius upper bound via the Chen & Kipping relation.  Default
            0.5, corresponding to an 86.6 % confidence-interval upper
            bound on the true mass under an isotropic inclination prior
            (Stevens & Gaudi 2013).
        """
        self.sources = sources
        self.msini_sin_min = msini_sin_min

    param_names_quantities = [
        'st_rad',
        'st_mass',
        'st_teff',
        'st_logg',
        'st_met',
        'st_lum',
        'sy_vmag',
        'sy_kmag',
        'sy_dist',
        'pl_insol',
        'pl_eqt',
    ]

    param_names_strings = [
        'st_spectype'
    ]

    param_names = param_names_quantities + param_names_strings


    def _merge_values(
        self,
        table: Table,
        params_q: dict[str, _ParamQtyArrays],
        params_s: dict[str, _ParamStrArrays],
        upper_error_suffix: str,
        lower_error_suffix: str,
        input_starname_key: str | None = None,
        id_supplier: IdSupplierBase | None = None,
        alternate_ids: Table | None = None,
    ) -> None:
        """Merge fundamental stellar and planetary values from parameter sources.

        Iterates over the table rows and polls configured sources in priority order to 
        fill missing parameters. Missing core stellar parameters (Teff, radius, mass) 
        are then derived sequentially per row using log(g) or estimation techniques.

        Parameters
        ----------
        table : Table
            The input catalog table.
        params_q : dict of str to _ParamQtyArrays
            Dictionary of quantitative parameters arrays to be filled in-place.
        params_s : dict of str to _ParamStrArrays
            Dictionary of string parameters arrays to be filled in-place.
        upper_error_suffix : str
            Suffix for upper error columns.
        lower_error_suffix : str
            Suffix for lower error columns.
        input_starname_key : str, optional
            Column name used to identify the host star for lookups.
        id_supplier : IdSupplierBase, optional
            An ID supplier for cross-referencing identifiers.
        alternate_ids : Table, optional
            A pre-loaded table of alternate IDs to use instead of fetching.
        """
        rad = params_q['st_rad']
        mass = params_q['st_mass']
        teff = params_q['st_teff']
        logg = params_q['st_logg']
        met = params_q['st_met']
        lum = params_q['st_lum']
        insol = params_q['pl_insol']
        vmag = params_q['sy_vmag']
        kmag = params_q['sy_kmag']
        dist = params_q['sy_dist']
        spec_arr = params_s['st_spectype']

        _SHORT = {
            'st_rad': 'rad', 'st_mass': 'mass', 'st_teff': 'teff',
            'st_logg': 'logg', 'st_met': 'met', 'st_lum': 'lum',
            'sy_vmag': 'vmag', 'sy_kmag': 'kmag', 'sy_dist': 'dist',
            'pl_insol': 'insol', 'pl_eqt': 'pl_eqt',
            'st_spectype': 'spec',
        }

        for i, row in enumerate(table):
            merged: dict = {}
            merged_src: dict = {}
            merged_err1: dict = {}
            merged_err2: dict = {}
            for source in self.sources:
                d = source.get(
                    row,
                    input_starname_key=input_starname_key,
                    id_supplier=id_supplier,
                    alternate_ids=alternate_ids,
                )
                for key, value in d.items():
                    if key.endswith(f'_{upper_error_suffix}') or key.endswith(f'_{lower_error_suffix}'):
                        continue
                    if key not in merged:
                        merged[key]  = value
                        merged_src[key]  = source.source_name
                        merged_err1[key] = d.get(f'{key}_{upper_error_suffix}')
                        merged_err2[key] = d.get(f'{key}_{lower_error_suffix}')

            def _bind(param):
                short = _SHORT.get(param, param)
                p = params_q[param] if param in params_q else params_s[param]
                if short in merged and p.mask[i]:
                    p.val[i]  = merged[short]
                    p.mask[i] = False
                    p.src[i]  = merged_src[short]
                    if param in params_q:
                        if merged_err1.get(short) is not None:
                            p.err1[i] = merged_err1[short]
                        if merged_err2.get(short) is not None:
                            p.err2[i] = merged_err2[short]
                    return True
                return False

            for param in self.param_names_quantities:
                _bind(param)
            for param in self.param_names_strings:
                _bind(param)

            # Teff derivation: if temperature not bound, attempt estimation using helpers or spec
            _derive_stellar_teff(i, teff, rad, lum, spec_arr)

            # Radius derivation: if radius not bound, attempt estimation using helpers
            _derive_stellar_radius(
                i, rad, mass, logg, teff, lum, met, kmag, dist, merged.get('spec') or ''
            )

            # Mass derivation from logg and radius
            _derive_stellar_mass(i, mass, rad, logg)

            # Re-bind remaining parameters that were skipped/derived
            for param in ('st_logg', 'st_mass', 'pl_insol', 'st_met', 'pl_eqt',
                          'sy_vmag', 'sy_kmag', 'sy_dist', 'st_lum'):
                _bind(param)

            if 'spec' in merged:
                spec_arr.val[i]     = merged['spec']
                spec_arr.src[i]     = merged_src.get('spec', '')

    def enrich(
        self,
        table: Table,
        planet_radius_key: str = 'r',
        planet_flux_key: str = None,
        planet_equilibrium_temperature_key: str = None,
        semi_major_axis_key: str = 'a',
        period_key: str = 'p',
        msini_key: str = 'msini',
        star_spectral_type_key = None,
        star_radius_key: str = None,
        star_mass_key: str = None,
        star_effective_temperature_key: str = None,
        star_logg_key: str = None,
        star_metallicity_key: str = None,
        star_luminosity_key: str = None,
        vmag_key: str = None,
        kmag_key: str = None,
        distance_key: str = None,
        upper_error_suffix: str | None = None,
        lower_error_suffix: str | None = None,
        input_starname_key: str | None = None,
        id_supplier: IdSupplierBase | None = None,
        alternate_ids: Table | None = None,
        **override_keys,
    ) -> Table:
        r"""Return a copy of table with enriched columns added/replaced.

        Dataflow and Priority:
        1. Values natively present in the input `table` columns always take the highest priority.
        2. For any parameter that is missing or masked, the pipeline iterates through the 
           configured list of parameter `sources` in priority order. The first source to 
           return a valid, non-null value for a specific parameter fills that gap.
        3. Once all sources are exhausted, any remaining gaps in fundamental stellar 
           parameters (e.g., mass, radius, effective temperature) are estimated using 
           established physical or empirical derivation formulas (e.g., log(g) physical 
           relations, Stefan-Boltzmann, or polynomial scaling laws based on Teff/magnitudes).
        
        This method applies several physical formulas to derive missing planetary 
        parameters:
        
        - Semi-major axis (Kepler's Third Law):
          
          $$ a = \left( M_{\star} \left( \frac{P}{365.25} \right)^2 \right)^{1/3} $$
          
        - Insolation Flux:
          
          $$ S = \frac{L}{a^2} \quad \text{or} \quad S = \left( \frac{T_{\text{eq}}}{254.793} \right)^4 $$
          
        - Equilibrium Temperature:
          
          $$ T_{\text{eq}} = 254.793 \cdot S^{1/4} $$

        Parameters
        ----------
        table : Table
            The input catalog table to enrich.
        planet_radius_key : str, optional
            Column name for planet radius (default: 'r').
        planet_flux_key : str, optional
            Column name for planet insolation flux.
        planet_equilibrium_temperature_key : str, optional
            Column name for planet equilibrium temperature.
        semi_major_axis_key : str, optional
            Column name for semi-major axis (default: 'a').
        period_key : str, optional
            Column name for orbital period (default: 'p').
        msini_key : str, optional
            Column name for minimum mass $M \sin i$ (default: 'msini').
        star_spectral_type_key : str, optional
            Column name for stellar spectral type.
        star_radius_key : str, optional
            Column name for stellar radius.
        star_mass_key : str, optional
            Column name for stellar mass.
        star_effective_temperature_key : str, optional
            Column name for stellar effective temperature.
        star_logg_key : str, optional
            Column name for stellar surface gravity.
        star_metallicity_key : str, optional
            Column name for stellar metallicity.
        star_luminosity_key : str, optional
            Column name for stellar luminosity.
        vmag_key : str, optional
            Column name for visual magnitude.
        kmag_key : str, optional
            Column name for K-band magnitude.
        distance_key : str, optional
            Column name for distance.
        upper_error_suffix : str, optional
            Suffix for upper error columns. Defaults to config enrichment value.
        lower_error_suffix : str, optional
            Suffix for lower error columns. Defaults to config enrichment value.
        input_starname_key : str, optional
            Column name used to identify the host star for lookups.
        id_supplier : IdSupplierBase, optional
            An ID supplier for cross-referencing identifiers.
        alternate_ids : Table, optional
            A pre-loaded table of alternate IDs to use instead of fetching.
        **override_keys
            Additional aliases mapped to parameter keys.

        Returns
        -------
        Table
            A new table containing the combined and physically derived parameters.
        """
        # Set default suffixes from config if not provided
        if upper_error_suffix is None:
            upper_error_suffix = config.enrichment['source_err_suffix']
        if lower_error_suffix is None:
            lower_error_suffix = config.enrichment['dependent_err_suffix']

        n = len(table)

        # ---------------------------------------------------------------------
        # 1 Fold and resolve all _key parameters (except input_starname_key)
        # ---------------------------------------------------------------------
        # Start with default column names
        resolved_cols = {
            'planet_radius': 'r',
            'planet_flux': 'pl_insol',
            'planet_equilibrium_temperature': 'pl_eqt',
            'semi_major_axis': 'a',
            'period': 'p',
            'msini': 'msini',
            'star_radius': 'st_rad',
            'star_mass': 'st_mass',
            'star_effective_temperature': 'st_teff',
            'star_logg': 'st_logg',
            'star_metallicity': 'st_met',
            'star_luminosity': 'st_lum',
            'vmag': 'sy_vmag',
            'kmag': 'sy_kmag',
            'distance': 'sy_dist',
            'star_spectral_type': 'st_spectype',
        }

        # Collect explicit named params
        provided_args = {
            'planet_radius': planet_radius_key,
            'planet_flux': planet_flux_key,
            'planet_equilibrium_temperature': planet_equilibrium_temperature_key,
            'semi_major_axis': semi_major_axis_key,
            'period': period_key,
            'msini': msini_key,
            'star_spectral_type': star_spectral_type_key,
            'star_radius': star_radius_key,
            'star_mass': star_mass_key,
            'star_effective_temperature': star_effective_temperature_key,
            'star_logg': star_logg_key,
            'star_metallicity': star_metallicity_key,
            'star_luminosity': star_luminosity_key,
            'vmag': vmag_key,
            'kmag': kmag_key,
            'distance': distance_key,
        }

        # Override with any explicit or custom keys passed
        for k, v in provided_args.items():
            if v is not None:
                resolved_cols[k] = v
        for k, v in override_keys.items():
            if v is not None and k.endswith('_key'):
                canonical_key = k[:-4]
                if canonical_key in resolved_cols:
                    resolved_cols[canonical_key] = v

        # Set up quantities and strings mapping
        params_q = {key: _ParamQtyArrays(n) for key in self.param_names_quantities}
        params_s = {key: _ParamStrArrays(n) for key in self.param_names_strings}

        internal_to_canonical = {
            'st_rad': 'star_radius',
            'st_mass': 'star_mass',
            'st_teff': 'star_effective_temperature',
            'st_logg': 'star_logg',
            'st_met': 'star_metallicity',
            'st_lum': 'star_luminosity',
            'sy_vmag': 'vmag',
            'sy_kmag': 'kmag',
            'sy_dist': 'distance',
            'pl_insol': 'planet_flux',
            'pl_eqt': 'planet_equilibrium_temperature',
            'st_spectype': 'star_spectral_type',
        }

        # Pre-load input table values as highest-priority source for each parameter
        for param_name in self.param_names_quantities:
            canonical_name = internal_to_canonical[param_name]
            col_name = resolved_cols[canonical_name]
            if col_name in table.colnames:
                vals, mask = _col_float(table, col_name)
                p = params_q[param_name]
                p.val[:] = vals
                p.mask[:] = mask
                p.src[:] = ['input'] * n

        for param_name in self.param_names_strings:
            canonical_name = internal_to_canonical[param_name]
            col_name = resolved_cols[canonical_name]
            if col_name in table.colnames:
                raw = table[col_name]
                p = params_s[param_name]
                p.val = list(raw)
                p.mask = np.array([v == '' or v is None for v in raw])
                p.src = ['input'] * n

        # Fill raw parameters from configured sources and derive fundamental stellar properties
        self._merge_values(
            table, params_q, params_s, upper_error_suffix, lower_error_suffix,
            input_starname_key, id_supplier, alternate_ids
        )

        # Restore shortcut references needed for remaining column-level computations
        rad = params_q['st_rad']
        mass = params_q['st_mass']
        teff = params_q['st_teff']
        spec_arr = params_s['st_spectype']

        st_spectype = [
            spectype_display(spec_arr.val[i], teff.val[i] if not teff.mask[i] else 0.0)
            for i in range(n)
        ]

        # ── r_valid: gates msini bounds (callers convert r → R_earth via R_JUP_TO_EARTH) ──────────
        r_vals, r_orig_mask = _col_float(table, resolved_cols['planet_radius'])
        r_valid = ~r_orig_mask & (r_vals > 0)

        # ── r_lower_bound / r_upper_bound (msini-based radius range) ─────────────
        r_lower_bound_arr = np.full(n, np.nan)
        r_upper_bound_arr = np.full(n, np.nan)
        msini_vals, msini_mask_arr = _col_float(table, resolved_cols['msini'])
        msini_earth = msini_vals * _M_JUP_TO_EARTH
        msini_active = ~msini_mask_arr & (msini_earth > 0) & ~r_valid
        for i in np.where(msini_active)[0]:
            r_lower_bound_arr[i] = mass_radius_chen_kipping(msini_earth[i])
            r_upper_bound_arr[i] = mass_radius_chen_kipping(
                msini_earth[i] / self.msini_sin_min)

        # ── st_lum: source-provided where available, else R²(T/T☉)⁴ ──────────
        src_lum = params_q['st_lum']
        computed_lum_mask = rad.mask | teff.mask
        computed_lum = np.where(~computed_lum_mask, rad.val ** 2 * (teff.val / T_SUN) ** 4, np.nan)

        lum      = np.where(~src_lum.mask, src_lum.val, computed_lum)
        lum_mask = src_lum.mask & computed_lum_mask
        st_lum_src = [
            src_lum.src[i] if not src_lum.mask[i] else
            (f"r:{rad.src[i]} teff:{teff.src[i]}" if not computed_lum_mask[i] else '')
            for i in range(n)
        ]

        # ── st_lum_err1/2 ─────────────────────────────────────────────────────
        def _rel(p):
            return (
                np.where(p.mask | ~(p.val > 0) | ~np.isfinite(p.err1), 0.0, p.err1 / p.val),
                np.where(p.mask | ~(p.val > 0) | ~np.isfinite(p.err2), 0.0, p.err2 / p.val),
            )

        rad_rel1, rad_rel2   = _rel(rad)
        teff_rel1, teff_rel2 = _rel(teff)

        lum_rel1_sq = (2.0 * rad_rel1) ** 2 + (4.0 * teff_rel1) ** 2
        lum_rel2_sq = (2.0 * rad_rel2) ** 2 + (4.0 * teff_rel2) ** 2
        has_prop_err1 = ~computed_lum_mask & (lum_rel1_sq > 0)
        has_prop_err2 = ~computed_lum_mask & (lum_rel2_sq > 0)
        with np.errstate(invalid='ignore'):
            lum_err1_prop = np.where(has_prop_err1, np.abs(computed_lum) * np.sqrt(lum_rel1_sq), np.nan)
            lum_err2_prop = np.where(has_prop_err2, np.abs(computed_lum) * np.sqrt(lum_rel2_sq), np.nan)

        lum_err1 = np.where(~src_lum.mask, src_lum.err1, lum_err1_prop)
        lum_err2 = np.where(~src_lum.mask, src_lum.err2, lum_err2_prop)

        # ── semi-major axis with Kepler fallback ──────────────────────────────
        a_vals, a_orig_mask = _col_float(table, resolved_cols['semi_major_axis'])
        p_vals, p_orig_mask = _col_float(table, resolved_cols['period'])
        mass_for_kepler = np.where(~mass.mask, mass.val, 1.0)
        kepler_cond = ~p_orig_mask & (p_vals > 0) & ~mass.mask
        a_kepler = np.where(
            kepler_cond,
            (mass_for_kepler * (p_vals / 365.25) ** 2) ** (1 / 3),
            0.0,
        )
        emc_a_valid = (~a_orig_mask) & (a_vals > 0)
        a_merged = np.where(emc_a_valid, a_vals, a_kepler)
        a_valid  = a_merged > 0

        a_src_list = [
            'provided' if emc_a_valid[i]
            else (f"kepler(mass:{mass.src[i]} p:{resolved_cols['period']})" if kepler_cond[i] else '')
            for i in range(n)
        ]

        # ── a_err1/2: Kepler only ─────────────────────────────────────────────
        kepler_base = kepler_cond & (mass_for_kepler > 0)
        a_err1 = np.where(kepler_base & np.isfinite(mass.err1),
                          (1.0/3.0) * a_kepler * mass.err1 / mass_for_kepler, np.nan)
        a_err2 = np.where(kepler_base & np.isfinite(mass.err2),
                          (1.0/3.0) * a_kepler * mass.err2 / mass_for_kepler, np.nan)

        # ── insolation flux (pl_insol) ────────────────────────────────────────
        insol = params_q['pl_insol']
        eqt = params_q['pl_eqt']
        with np.errstate(divide='ignore', invalid='ignore'):
            flux_calc = np.where(a_valid, lum / a_merged ** 2, np.nan)
        
        flux_eqt_calc = np.where(~eqt.mask & (eqt.val > 0), (eqt.val / 254.793) ** 4, np.nan)
        
        flux_final = np.where(~insol.mask, insol.val, np.where(np.isfinite(flux_calc), flux_calc, flux_eqt_calc))
        flux_mask  = ~np.isfinite(flux_final)

        flux_rel_src = []
        for i in range(n):
            if not insol.mask[i]:
                flux_rel_src.append(f"insol:{insol.src[i]}")
            elif np.isfinite(flux_calc[i]):
                flux_rel_src.append(f"r:{rad.src[i]} teff:{teff.src[i]} a:{a_src_list[i]}")
            elif np.isfinite(flux_eqt_calc[i]):
                flux_rel_src.append(f"derived(eqt:{eqt.src[i]})")
            else:
                flux_rel_src.append('')

        computed_case = insol.mask & np.isfinite(flux_calc) & ~lum_mask & (lum != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lum_up_sq = np.where(np.isfinite(lum_err1) & computed_case, (lum_err1 / lum) ** 2, 0.0)
            a_down_sq = np.where(computed_case & np.isfinite(a_err2) & (a_merged > 0), (2.0 * a_err2 / a_merged) ** 2, 0.0)
            lum_dn_sq = np.where(np.isfinite(lum_err2) & computed_case, (lum_err2 / lum) ** 2, 0.0)
            a_up_sq   = np.where(computed_case & np.isfinite(a_err1) & (a_merged > 0), (2.0 * a_err1 / a_merged) ** 2, 0.0)

        err1_sq = lum_up_sq + a_down_sq
        err2_sq = lum_dn_sq + a_up_sq
        
        eqt_case = insol.mask & ~np.isfinite(flux_calc) & np.isfinite(flux_eqt_calc)
        with np.errstate(divide='ignore', invalid='ignore'):
            eqt_err1_prop = np.where(eqt_case & np.isfinite(eqt.err1) & (eqt.val > 0), 4.0 * flux_eqt_calc * eqt.err1 / eqt.val, np.nan)
            eqt_err2_prop = np.where(eqt_case & np.isfinite(eqt.err2) & (eqt.val > 0), 4.0 * flux_eqt_calc * eqt.err2 / eqt.val, np.nan)

        # Populate final values/errors in-place
        insol.val[:] = flux_final
        insol.mask[:] = flux_mask
        insol.src[:] = flux_rel_src
        
        err1_calc = np.where(err1_sq > 0, np.abs(flux_calc) * np.sqrt(err1_sq), np.nan)
        err2_calc = np.where(err2_sq > 0, np.abs(flux_calc) * np.sqrt(err2_sq), np.nan)
        
        insol.err1[:] = np.where(~insol.mask, insol.err1, np.where(np.isfinite(err1_calc), err1_calc, eqt_err1_prop))
        insol.err2[:] = np.where(~insol.mask, insol.err2, np.where(np.isfinite(err2_calc), err2_calc, eqt_err2_prop))

        # ── planet equilibrium temperature (pl_eqt) fallback from insolation ──
        eqt = params_q['pl_eqt']
        eqt_calc = 254.793 * (insol.val ** 0.25)
        
        eqt_mask = eqt.mask & ~insol.mask & np.isfinite(eqt_calc) & (eqt_calc > 0)
        
        eqt.val[eqt_mask] = eqt_calc[eqt_mask]
        for idx in np.where(eqt_mask)[0]:
            eqt.src[idx] = f"derived(insol:{insol.src[idx]})"
            s_val = insol.val[idx]
            t_val = eqt_calc[idx]
            if s_val > 0:
                if np.isfinite(insol.err1[idx]):
                    eqt.err1[idx] = 0.25 * t_val * insol.err1[idx] / s_val
                if np.isfinite(insol.err2[idx]):
                    eqt.err2[idx] = 0.25 * t_val * insol.err2[idx] / s_val
        
        eqt.mask[:] = eqt.mask & ~eqt_mask

        spectral_category = [classify_spectral_type(s) for s in st_spectype]
        spec_src = [spec_arr.src[i] for i in range(n)]

        def _mc(arr, name, desc=''):
            return MaskedColumn(arr, mask=~np.isfinite(arr), name=name, description=desc)

        result = table.copy()
        for default_col, key, desc in (
            ('st_teff', 'st_teff',   ''),
            ('st_rad',  'st_rad',    ''),
            ('st_mass', 'st_mass',   ''),
            ('st_logg', 'st_logg',   'Stellar log surface gravity [log(cm/s^2)]'),
            ('st_met',  'st_met',    'Stellar metallicity [Fe/H]'),
            ('st_lum',  'st_lum',    'Stellar luminosity [L_sun]'),
            ('sy_vmag', 'sy_vmag',   ''),
            ('sy_kmag', 'sy_kmag',   '2MASS K-band magnitude [mag]'),
            ('sy_dist', 'sy_dist',   ''),
            ('pl_eqt',  'pl_eqt',    'Planet equilibrium temperature [K]'),
            ('pl_insol', 'pl_insol', 'Planet insolation flux [S_earth]'),
        ):
            canonical_name = internal_to_canonical[key]
            col = resolved_cols[canonical_name]
            p = params_q[key]
            result[col] = MaskedColumn(p.val, mask=p.mask, name=col, description=desc)
            result[f'{col}_src'] = p.src
            result[f'{col}_{upper_error_suffix}'] = _mc(p.err1, f'{col}_{upper_error_suffix}')
            result[f'{col}_{lower_error_suffix}'] = _mc(p.err2, f'{col}_{lower_error_suffix}')

        result['st_spectype']        = st_spectype
        result['st_spectype_src']    = spec_src
        result['st_lum']             = MaskedColumn(lum,    mask=lum_mask,     name='st_lum',
                                                    description='Stellar luminosity [L_sun]')
        result[f'st_lum_{upper_error_suffix}']        = _mc(lum_err1,   f'st_lum_{upper_error_suffix}',  'Luminosity upper 1σ [L_sun]')
        result[f'st_lum_{lower_error_suffix}']        = _mc(lum_err2,   f'st_lum_{lower_error_suffix}',  'Luminosity lower 1σ [L_sun]')
        result['st_lum_src']         = st_lum_src
        result['r_lower_bound']      = _mc(r_lower_bound_arr, 'r_lower_bound',
                                           'Min estimated planet radius from msini [R_earth]')
        result['r_upper_bound']      = _mc(r_upper_bound_arr, 'r_upper_bound',
                                           'Max estimated planet radius from msini [R_earth]')
        result['a_src']              = a_src_list
        result['spectral_category']  = spectral_category
        return result
