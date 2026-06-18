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
    mass_radius_chen_kipping,
    T_SUN,
)
from crossmatching.enrichment.inference import (
    infer_star_teff,
    infer_star_radius,
    infer_star_mass,
    ParamQty,
    ParamStr,
)

_M_JUP_TO_EARTH = u.M_jup.to(u.M_earth)  # ~317.83
R_JUP_TO_EARTH  = u.R_jup.to(u.R_earth)  # ~11.21; exported for callers that need r in R_earth


def _to_float(val) -> float:
    """Safely convert value to standard float, replacing masked/None with NaN."""
    if val is None or np.ma.is_masked(val):
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan
    
def _mc(arr, name, desc=''):
    return MaskedColumn(arr, mask=~np.isfinite(arr), name=name, description=desc)

def _try_get_column(table: Table, col: str):
    """Return (values, mask) float arrays for a table column.

    Returns an all-masked pair when the column is absent, so the
    optional columns degrade gracefully on tables with other layouts.

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
    
    return table[col], np.ma.getmaskarray(table[col])

def _extract_param_arrays(param_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Extract physical parameter attributes from a list of objects into vectorized arrays.

    Parameters
    ----------
    param_list : list
        A list of parameter objects (e.g., ParamQty) containing the attributes 
        'val', 'mask', 'err1', 'err2', and 'src'.

    Returns
    -------
    val : np.ndarray
        Array of parameter values.
    mask : np.ndarray
        Boolean array indicating masked or invalid entries.
    err1 : np.ndarray
        Array of upper error values.
    err2 : np.ndarray
        Array of lower error values.
    src : list
        List of data source strings for each entry.
    """
    return (
        np.array([x.val for x in param_list]),
        np.array([x.mask for x in param_list]),
        np.array([x.err1 for x in param_list]),
        np.array([x.err2 for x in param_list]),
        [x.src for x in param_list]
    )

        
class ParamFiller:
    """Enriches a catalog table with stellar and derived planetary parameters.

    Applies a priority-ordered chain of ParamSource objects: for each
    row the first source that provides a value for a given parameter wins.
    Computes derived physical columns (r_lower_bound, r_upper_bound, a, pl_insol, spectral_category)
    from the merged parameters.
    """


    PARAM_NAMES_QUANTITIES = [
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

    PARAM_NAMES_STRINGS = [
        'st_spectype'
    ]

    PARAM_NAMES = PARAM_NAMES_QUANTITIES + PARAM_NAMES_STRINGS

    PARAM_METADATA = {
        'st_rad':      ('star_radius', 'Stellar radius [R_sun]'),
        'st_mass':     ('star_mass', 'Stellar mass [M_sun]'),
        'st_teff':     ('star_effective_temperature', 'Stellar effective temperature [K]'),
        'st_logg':     ('star_logg', 'Stellar log surface gravity [log(cm/s^2)]'),
        'st_met':      ('star_metallicity', 'Stellar metallicity [Fe/H]'),
        'st_lum':      ('star_luminosity', 'Stellar luminosity [L_sun]'),
        'sy_vmag':     ('vmag', 'Visual band magnitude [mag]'),
        'sy_kmag':     ('kmag', '2MASS K-band magnitude [mag]'),
        'sy_dist':     ('distance', 'Distance [pc]'),
        'pl_insol':    ('planet_flux', 'Planet insolation flux [S_earth]'),
        'pl_eqt':      ('planet_equilibrium_temperature', 'Planet equilibrium temperature [K]'),
        'st_spectype': ('star_spectral_type', 'Stellar spectral type')
    }
    

    def __init__(self, sources: list[ParamSource], msini_sin_min: float = 0.3):
        """Initialise the merger with an ordered list of parameter sources.

        Parameters
        ----------
        sources : list of `~crossmatching.param_sources.base.ParamSource`
            Priority-ordered    parameter sources.  For each  parameter the 
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


    def _populate_quantity_columns(
        self,
        result_table,
        param_names: list[str], 
        params_q: dict, 
        resolved_cols: dict, 
        upper_suffix: str, 
        lower_suffix: str
    ) -> None: 
        """
        Dynamically populate an Astropy table with quantity columns and their errors.

        Parameters
        ----------
        result_table : Table
            The Astropy table to be enriched in-place.
        param_names : list
            List of parameter keys (e.g., 'st_rad') to process.
        params_q : dict
            Dictionary mapping parameter keys to lists of parameter objects.
        resolved_cols : dict
            Mapping of canonical parameter names to actual column names in the table.
        upper_suffix : str
            Suffix to append to the upper error column name.
        lower_suffix : str
            Suffix to append to the lower error column name.

        Returns
        -------
        None
            The `result_table` is modified in-place.
        """

        for key in param_names:
            canonical_name, desc = self.PARAM_METADATA[key]
            col = resolved_cols[canonical_name]
            
            p_val, p_mask, p_err1, p_err2, p_src = _extract_param_arrays(params_q[key])
            
            result_table[col] = MaskedColumn(p_val, mask=p_mask, name=col, description=desc)
            result_table[f'{col}_src'] = p_src
            result_table[f'{col}_{upper_suffix}'] = _mc(p_err1, f'{col}_{upper_suffix}')
            result_table[f'{col}_{lower_suffix}'] = _mc(p_err2, f'{col}_{lower_suffix}')
        

    def _merge_values(
        self,
        table: Table,
        params_q: dict[str, list[ParamQty]],
        params_s: dict[str, list[ParamStr]],
        upper_error_suffix: str,
        lower_error_suffix: str,
        input_starname_key: str | None = None,
        id_supplier: IdSupplierBase | None = None,
        alternate_ids: Table | None = None,
        disable_calculations: bool = False,
    ) -> None:
        """Merge fundamental stellar and planetary values from parameter sources.

        Iterates over the table rows and polls configured sources in priority order to 
        fill missing parameters. Missing core stellar parameters (Teff, radius, mass) 
        are then derived sequentially per row using log(g) or estimation techniques.

        Parameters
        ----------
        table : Table
            The input catalog table.
        params_q : dict of str to list of ParamQty
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
                data = source.get(
                    row,
                    input_starname_key=input_starname_key,
                    id_supplier=id_supplier,
                    alternate_ids=alternate_ids,
                )
                for key, value in data.items():
                    if key.endswith(f'_{upper_error_suffix}') or key.endswith(f'_{lower_error_suffix}'):
                        continue
                    if key not in merged:
                        merged[key]  = value
                        merged_src[key]  = source.source_name
                        merged_err1[key] = data.get(f'{key}_{upper_error_suffix}')
                        merged_err2[key] = data.get(f'{key}_{lower_error_suffix}')

            def _bind(param):
                short = _SHORT.get(param, param)
                if short in merged:
                    if param in params_q:
                        pq = params_q[param][i]
                        if pq.mask:
                            pq.val  = _to_float(merged[short])
                            pq.mask = False
                            pq.src  = merged_src[short]
                            if merged_err1.get(short) is not None:
                                pq.err1 = _to_float(merged_err1[short])
                            if merged_err2.get(short) is not None:
                                pq.err2 = _to_float(merged_err2[short])
                            return True
                    else:
                        ps = params_s[param][i]
                        if ps.mask:
                            ps.val  = str(merged[short]) if merged[short] is not None else ""
                            ps.mask = False
                            ps.src  = merged_src[short]
                            return True
                return False

            for param in self.PARAM_NAMES_QUANTITIES:
                _bind(param)
            for param in self.PARAM_NAMES_STRINGS:
                _bind(param)

            if not disable_calculations:
                # Teff derivation: if temperature not bound, attempt estimation using helpers or spec
                spec_val = merged.get('spec')
                spec_src = merged_src.get('spec')
                if not spec_val and not spec_arr[i].mask:
                    spec_val = spec_arr[i].val
                    spec_src = spec_arr[i].src
                if not spec_val:
                    spec_val = ""
                    spec_src = ""

                spectype_obj = ParamStr(
                    val=spec_val,
                    src=spec_src,
                    mask=not spec_val
                )

                params_q['st_teff'][i] = infer_star_teff(
                    teff[i], rad[i], lum[i], spectype_obj
                )

                # Radius derivation: if radius not bound, attempt estimation using helpers
                params_q['st_rad'][i] = infer_star_radius(
                    rad[i], mass[i], logg[i], teff[i], lum[i], met[i], kmag[i], dist[i], spectype_obj
                )

                # Mass derivation from logg and radius
                params_q['st_mass'][i] = infer_star_mass(
                    mass[i], rad[i], logg[i]
                )

            # Re-bind remaining parameters that were skipped/derived
            for param in ('st_logg', 'st_mass', 'pl_insol', 'st_met', 'pl_eqt',
                          'sy_vmag', 'sy_kmag', 'sy_dist', 'st_lum'):
                _bind(param)

            if 'spec' in merged:
                spec_arr[i].val  = str(merged['spec']) if merged['spec'] is not None else ""
                spec_arr[i].mask = False
                spec_arr[i].src  = merged_src.get('spec', '')

    def enrich(
        self,
        table: Table,
        planet_radius_key: str = None,
        planet_flux_key: str = None,
        planet_equilibrium_temperature_key: str = None,
        semi_major_axis_key: str = None,
        period_key: str = None,
        msini_key: str = None,
        star_spectral_type_key: str = None,
        star_radius_key: str = None,
        star_mass_key: str = None,
        star_effective_temperature_key: str = None,
        star_logg_key: str = None,
        star_metallicity_key: str = None,
        star_luminosity_key: str = None,
        vmag_key: str = None,
        kmag_key: str = None,
        distance_key: str = None,
        upper_error_suffix: str  = "err1",
        lower_error_suffix: str  = "err2",
        input_starname_key: str | None = None,
        id_supplier: IdSupplierBase | None = None,
        alternate_ids: Table | None = None,
        disable_calculations: bool = False,
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
            Column name for planet radius 
        planet_flux_key : str, optional
            Column name for planet insolation flux.
        planet_equilibrium_temperature_key : str, optional
            Column name for planet equilibrium temperature.
        semi_major_axis_key : str, optional
            Column name for semi-major axis
        period_key : str, optional
            Column name for orbital period 
        msini_key : str, optional
            Column name for minimum mass $M \sin i$
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
        upper_error_suffix : str = "err1"
            Suffix for upper error columns. Defaults to config enrichment value.
        lower_error_suffix : str = "err2"
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
        n = len(table)

        # NOTE: while these dicts are quite verbose,
        # they are explicitely stated to help with IDE autocompletion when accesing the resulting table
        resolved_cols = {
            'planet_radius': 'pl_rad',
            'planet_flux': 'pl_insol',
            'planet_equilibrium_temperature': 'pl_eqt',
            'semi_major_axis': 'semi_major_axis',
            'period': 'period',
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

        # Set up quantities and strings mapping
        params_q = {key: [ParamQty() for _ in range(n)] for key in self.PARAM_NAMES_QUANTITIES}
        params_s = {key: [ParamStr() for _ in range(n)] for key in self.PARAM_NAMES_STRINGS}

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
        for param_name in self.PARAM_NAMES_QUANTITIES:
            canonical_name = internal_to_canonical[param_name]
            col_name = resolved_cols[canonical_name]
            if col_name in table.colnames:
                vals, mask = _try_get_column(table, col_name)
                p = params_q[param_name]
                for idx, (val, msk) in enumerate(zip(vals, mask)):
                    p[idx].val = _to_float(val)
                    p[idx].mask = bool(msk)
                    p[idx].src = 'input'

        for param_name in self.PARAM_NAMES_STRINGS:
            canonical_name = internal_to_canonical[param_name]
            col_name = resolved_cols[canonical_name]
            if col_name in table.colnames:
                raw = table[col_name]
                p = params_s[param_name]
                for idx, val in enumerate(raw):
                    p[idx].val = str(val) if val is not None else ""
                    p[idx].mask = val == '' or val is None
                    p[idx].src = 'input'

        # Fill raw parameters from configured sources and derive fundamental stellar properties
        self._merge_values(
            table, params_q, params_s, upper_error_suffix, lower_error_suffix,
            input_starname_key, id_supplier, alternate_ids, disable_calculations
        )

        result = table.copy()
        if disable_calculations:
            self._populate_quantity_columns(result, self.PARAM_NAMES_QUANTITIES, params_q, resolved_cols, upper_error_suffix, lower_error_suffix)
            for key in self.PARAM_NAMES_STRINGS:
                col = resolved_cols[self.PARAM_METADATA[key][0]]
                result[col] = [x.val for x in params_s[key]]
                result[f'{col}_src'] = [x.src for x in params_s[key]]
            return result

        # Convert lists of ParamQty to numpy arrays for vectorized calculations
        rad_val, rad_mask, rad_err1, rad_err2, rad_src = _extract_param_arrays(params_q['st_rad'])
        mass_val, mass_mask, mass_err1, mass_err2, mass_src = _extract_param_arrays(params_q['st_mass'])
        teff_val, teff_mask, teff_err1, teff_err2, teff_src = _extract_param_arrays(params_q['st_teff'])
        lum_val, lum_mask, lum_err1, lum_err2, lum_src = _extract_param_arrays(params_q['st_lum'])

        spec_arr = params_s['st_spectype']
        st_spectype = [spectype_display(spec_arr[i].val, teff_val[i] if not teff_mask[i] else 0.0) for i in range(n)]

        # ── r_valid: gates msini bounds (callers convert r → R_earth via R_JUP_TO_EARTH) ──────────
        r_vals, r_orig_mask = _try_get_column(table, resolved_cols['planet_radius'])
        r_valid = ~r_orig_mask & (r_vals > 0)

        # ── r_lower_bound / r_upper_bound (msini-based radius range) ─────────────
        r_lower_bound_arr = np.full(n, np.nan)
        r_upper_bound_arr = np.full(n, np.nan)
        msini_vals, msini_mask_arr = _try_get_column(table, resolved_cols['msini'])
        msini_earth = msini_vals * _M_JUP_TO_EARTH
        msini_active = ~msini_mask_arr & (msini_earth > 0) & ~r_valid
        for i in np.where(msini_active)[0]:
            r_lower_bound_arr[i] = mass_radius_chen_kipping(msini_earth[i])
            r_upper_bound_arr[i] = mass_radius_chen_kipping(
                msini_earth[i] / self.msini_sin_min)

        # ── st_lum: source-provided where available, else R²(T/T☉)⁴ ──────────
        computed_lum_mask = rad_mask | teff_mask
        computed_lum = np.where(~computed_lum_mask, rad_val ** 2 * (teff_val / T_SUN) ** 4, np.nan)

        lum      = np.where(~lum_mask, lum_val, computed_lum)
        lum_mask = lum_mask & computed_lum_mask
        st_lum_src = [
            lum_src[i] if not lum_mask[i] else
            (f"r:{rad_src[i]} teff:{teff_src[i]}" if not computed_lum_mask[i] else '')
            for i in range(n)
        ]

        # ── st_lum_err1/2 ─────────────────────────────────────────────────────
        def _rel(p_mask, p_val, p_err1, p_err2):
            return (
                np.where(p_mask | ~(p_val > 0) | ~np.isfinite(p_err1), 0.0, p_err1 / p_val),
                np.where(p_mask | ~(p_val > 0) | ~np.isfinite(p_err2), 0.0, p_err2 / p_val),
            )

        rad_rel1, rad_rel2   = _rel(rad_mask, rad_val, rad_err1, rad_err2)
        teff_rel1, teff_rel2 = _rel(teff_mask, teff_val, teff_err1, teff_err2)

        lum_rel1_sq = (2.0 * rad_rel1) ** 2 + (4.0 * teff_rel1) ** 2
        lum_rel2_sq = (2.0 * rad_rel2) ** 2 + (4.0 * teff_rel2) ** 2
        has_prop_err1 = ~computed_lum_mask & (lum_rel1_sq > 0)
        has_prop_err2 = ~computed_lum_mask & (lum_rel2_sq > 0)
        with np.errstate(invalid='ignore'):
            lum_err1_prop = np.where(has_prop_err1, np.abs(computed_lum) * np.sqrt(lum_rel1_sq), np.nan)
            lum_err2_prop = np.where(has_prop_err2, np.abs(computed_lum) * np.sqrt(lum_rel2_sq), np.nan)

        lum_err1 = np.where(~lum_mask, lum_err1, lum_err1_prop)
        lum_err2 = np.where(~lum_mask, lum_err2, lum_err2_prop)

        # ── semi-major axis with Kepler fallback ──────────────────────────────
        a_vals, a_orig_mask = _try_get_column(table, resolved_cols['semi_major_axis'])
        p_vals, p_orig_mask = _try_get_column(table, resolved_cols['period'])
        mass_for_kepler = np.where(~mass_mask, mass_val, 1.0)
        kepler_cond = ~p_orig_mask & (p_vals > 0) & ~mass_mask
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
            else (f"kepler(mass:{mass_src[i]} p:{resolved_cols['period']})" if kepler_cond[i] else '')
            for i in range(n)
        ]

        # ── a_err1/2: Kepler only ─────────────────────────────────────────────
        kepler_base = kepler_cond & (mass_for_kepler > 0)
        a_err1 = np.where(kepler_base & np.isfinite(mass_err1),
                          (1.0/3.0) * a_kepler * mass_err1 / mass_for_kepler, np.nan)
        a_err2 = np.where(kepler_base & np.isfinite(mass_err2),
                          (1.0/3.0) * a_kepler * mass_err2 / mass_for_kepler, np.nan)

        # ── insolation flux (pl_insol) ────────────────────────────────────────
        insol = params_q['pl_insol']
        eqt = params_q['pl_eqt']

        insol_mask = np.array([x.mask for x in insol])
        insol_val = np.array([x.val for x in insol])
        insol_src = [x.src for x in insol]
        insol_err1 = np.array([x.err1 for x in insol])
        insol_err2 = np.array([x.err2 for x in insol])

        eqt_mask_arr = np.array([x.mask for x in eqt])
        eqt_val_arr = np.array([x.val for x in eqt])
        eqt_src_arr = [x.src for x in eqt]

        with np.errstate(divide='ignore', invalid='ignore'):
            flux_calc = np.where(a_valid, lum / a_merged ** 2, np.nan)
        
        flux_eqt_calc = np.where(~eqt_mask_arr & (eqt_val_arr > 0), (eqt_val_arr / 254.793) ** 4, np.nan)
        
        flux_final = np.where(~insol_mask, insol_val, np.where(np.isfinite(flux_calc), flux_calc, flux_eqt_calc))
        flux_mask  = ~np.isfinite(flux_final)

        flux_rel_src = []
        for i in range(n):
            if not insol_mask[i]:
                flux_rel_src.append(f"insol:{insol_src[i]}")
            elif np.isfinite(flux_calc[i]):
                flux_rel_src.append(f"r:{rad_src[i]} teff:{teff_src[i]} a:{a_src_list[i]}")
            elif np.isfinite(flux_eqt_calc[i]):
                flux_rel_src.append(f"derived(eqt:{eqt_src_arr[i]})")
            else:
                flux_rel_src.append('')

        computed_case = insol_mask & np.isfinite(flux_calc) & ~lum_mask & (lum != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lum_up_sq = np.where(np.isfinite(lum_err1) & computed_case, (lum_err1 / lum) ** 2, 0.0)
            a_down_sq = np.where(computed_case & np.isfinite(a_err2) & (a_merged > 0), (2.0 * a_err2 / a_merged) ** 2, 0.0)
            lum_dn_sq = np.where(np.isfinite(lum_err2) & computed_case, (lum_err2 / lum) ** 2, 0.0)
            a_up_sq   = np.where(computed_case & np.isfinite(a_err1) & (a_merged > 0), (2.0 * a_err1 / a_merged) ** 2, 0.0)

        err1_sq = lum_up_sq + a_down_sq
        err2_sq = lum_dn_sq + a_up_sq
        
        eqt_case = insol_mask & ~np.isfinite(flux_calc) & np.isfinite(flux_eqt_calc)
        eqt_err1_arr = np.array([x.err1 for x in eqt])
        eqt_err2_arr = np.array([x.err2 for x in eqt])
        with np.errstate(divide='ignore', invalid='ignore'):
            eqt_err1_prop = np.where(eqt_case & np.isfinite(eqt_err1_arr) & (eqt_val_arr > 0), 4.0 * flux_eqt_calc * eqt_err1_arr / eqt_val_arr, np.nan)
            eqt_err2_prop = np.where(eqt_case & np.isfinite(eqt_err2_arr) & (eqt_val_arr > 0), 4.0 * flux_eqt_calc * eqt_err2_arr / eqt_val_arr, np.nan)

        err1_calc = np.where(err1_sq > 0, np.abs(flux_calc) * np.sqrt(err1_sq), np.nan)
        err2_calc = np.where(err2_sq > 0, np.abs(flux_calc) * np.sqrt(err2_sq), np.nan)
        
        final_insol_err1 = np.where(~insol_mask, insol_err1, np.where(np.isfinite(err1_calc), err1_calc, eqt_err1_prop))
        final_insol_err2 = np.where(~insol_mask, insol_err2, np.where(np.isfinite(err2_calc), err2_calc, eqt_err2_prop))

        # Populate final values/errors in-place in the insol objects
        for idx in range(n):
            insol[idx].val = flux_final[idx]
            insol[idx].mask = flux_mask[idx]
            insol[idx].src = flux_rel_src[idx]
            insol[idx].err1 = final_insol_err1[idx]
            insol[idx].err2 = final_insol_err2[idx]

        # ── planet equilibrium temperature (pl_eqt) fallback from insolation ──
        insol_final_val = np.array([x.val for x in insol])
        insol_final_mask = np.array([x.mask for x in insol])
        insol_final_err1 = np.array([x.err1 for x in insol])
        insol_final_err2 = np.array([x.err2 for x in insol])

        eqt_calc = 254.793 * (insol_final_val ** 0.25)
        
        eqt_mask = eqt_mask_arr & ~insol_final_mask & np.isfinite(eqt_calc) & (eqt_calc > 0)
        
        for idx in np.where(eqt_mask)[0]:
            eqt[idx].val = eqt_calc[idx]
            eqt[idx].src = f"derived(insol:{insol[idx].src})"
            s_val = insol_final_val[idx]
            t_val = eqt_calc[idx]
            if s_val > 0:
                if np.isfinite(insol_final_err1[idx]):
                    eqt[idx].err1 = 0.25 * t_val * insol_final_err1[idx] / s_val
                if np.isfinite(insol_final_err2[idx]):
                    eqt[idx].err2 = 0.25 * t_val * insol_final_err2[idx] / s_val
            eqt[idx].mask = False

        spectral_category = [classify_spectral_type(s) for s in st_spectype]
        spec_src = [spec_arr[i].src for i in range(n)]

        self._populate_quantity_columns(
            result, self.PARAM_NAMES_QUANTITIES, params_q, resolved_cols, 
            upper_error_suffix, lower_error_suffix
        )        
        
        def _mc(arr, name, desc=''):
            return MaskedColumn(arr, mask=~np.isfinite(arr), name=name, description=desc)


        result['st_spectype']        = st_spectype
        result['st_spectype_src']    = spec_src
        result['st_lum']             = MaskedColumn(lum,    mask=lum_mask,     name='st_lum',
                                                    description='Stellar luminosity [L_sun]')
        result[f'st_lum_{upper_error_suffix}']  = _mc(lum_err1,   f'st_lum_{upper_error_suffix}',  'Luminosity upper 1σ [L_sun]')
        result[f'st_lum_{lower_error_suffix}']  = _mc(lum_err2,   f'st_lum_{lower_error_suffix}',  'Luminosity lower 1σ [L_sun]')
        result['st_lum_src']         = st_lum_src
        result['r_lower_bound']      = _mc(r_lower_bound_arr, 'r_lower_bound',
                                           'Min estimated planet radius from msini [R_earth]')
        result['r_upper_bound']      = _mc(r_upper_bound_arr, 'r_upper_bound',
                                           'Max estimated planet radius from msini [R_earth]')
        result['a_src']              = a_src_list
        result['spectral_category']  = spectral_category
        return result
