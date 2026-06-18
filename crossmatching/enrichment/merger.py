from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.table import MaskedColumn, Table

from crossmatching import IdSupplierBase
from crossmatching.enrichment.param_sources.base import ParamSource
from crossmatching.enrichment.spectral_types import classify_spectral_type

from crossmatching.enrichment.inference import (
    infer_star_teff,
    infer_star_radius,
    infer_star_mass,
    infer_stellar_luminosity,
    infer_semi_major_axis,
    infer_msini_radius_bounds,
    infer_planet_insolation,
    infer_planet_equilibrium_temperature,
    infer_spectral_type_display,
    ParamQty,
    ParamStr,
)


def _to_float(val) -> float:
    """Safely convert value to float, replacing masked/None/invalid with NaN."""
    if val is None or np.ma.is_masked(val):
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _mc(values, name: str, desc: str = ""):
    arr = np.array(values, dtype=float)
    return MaskedColumn(arr, mask=~np.isfinite(arr), name=name, description=desc)


def _table_value_is_masked(table: Table, col: str, idx: int) -> bool:
    if col not in table.colnames:
        return True
    return bool(np.ma.getmaskarray(table[col])[idx])


def _qty_from_table(table: Table, col: str, idx: int, *, src: str = "input") -> ParamQty:
    if col not in table.colnames:
        return ParamQty()

    if _table_value_is_masked(table, col, idx):
        return ParamQty()

    q = ParamQty()
    q.val = _to_float(table[col][idx])
    q.err1 = np.nan
    q.err2 = np.nan
    q.src = src
    q.mask = not np.isfinite(q.val)

    return q


def _str_from_table(table: Table, col: str, idx: int, *, src: str = "input") -> ParamStr:
    if col not in table.colnames:
        return ParamStr()

    if _table_value_is_masked(table, col, idx):
        return ParamStr()

    s = ParamStr()
    s.val = "" if table[col][idx] is None else str(table[col][idx])
    s.src = src
    s.mask = not bool(s.val.strip())

    return s

def _put_qty_column(
    result: Table,
    col: str,
    params: list[ParamQty],
    *,
    description: str = "",
    upper_error_suffix: str = "err1",
    lower_error_suffix: str = "err2",
) -> None:
    values = [p.val for p in params]
    masks = [p.mask or not np.isfinite(p.val) for p in params]

    result[col] = MaskedColumn(
        values,
        mask=masks,
        name=col,
        description=description,
    )
    result[f"{col}_src"] = [p.src for p in params]
    result[f"{col}_{upper_error_suffix}"] = _mc(
        [p.err1 for p in params],
        f"{col}_{upper_error_suffix}",
    )
    result[f"{col}_{lower_error_suffix}"] = _mc(
        [p.err2 for p in params],
        f"{col}_{lower_error_suffix}",
    )


def _put_str_column(
    result: Table,
    col: str,
    params: list[ParamStr],
    *,
    display_values: list[str] | None = None,
) -> None:
    values = display_values if display_values is not None else [p.val for p in params]
    masks = [p.mask for p in params]

    result[col] = MaskedColumn(values, mask=masks, name=col)
    result[f"{col}_src"] = [p.src for p in params]


class ParamFiller:
    """Enriches a catalog table with stellar and derived planetary parameters."""

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
        "st_spectype",
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

    SOURCE_KEY_ALIASES = {
        "st_rad": "rad",
        "st_mass": "mass",
        "st_teff": "teff",
        "st_logg": "logg",
        "st_met": "met",
        "st_lum": "lum",
        "sy_vmag": "vmag",
        "sy_kmag": "kmag",
        "sy_dist": "dist",
        "pl_insol": "insol",
        "pl_eqt": "pl_eqt",
        "st_spectype": "spec",
    }

    def __init__(self, sources: list[ParamSource], msini_sin_min: float = 0.3):
        self.sources = sources
        self.msini_sin_min = msini_sin_min

    def _resolved_columns(
        self,
        *,
        planet_radius_key: str | None,
        planet_flux_key: str | None,
        planet_equilibrium_temperature_key: str | None,
        semi_major_axis_key: str | None,
        period_key: str | None,
        msini_key: str | None,
        star_spectral_type_key: str | None,
        star_radius_key: str | None,
        star_mass_key: str | None,
        star_effective_temperature_key: str | None,
        star_logg_key: str | None,
        star_metallicity_key: str | None,
        star_luminosity_key: str | None,
        vmag_key: str | None,
        kmag_key: str | None,
        distance_key: str | None,
        **override_keys
    ) -> dict[str, str]:
        resolved = {
            "planet_radius": "pl_rad",
            "planet_flux": "pl_insol",
            "planet_equilibrium_temperature": "pl_eqt",
            "semi_major_axis": "semi_major_axis",
            "period": "period",
            "msini": "msini",
            "star_radius": "st_rad",
            "star_mass": "st_mass",
            "star_effective_temperature": "st_teff",
            "star_logg": "st_logg",
            "star_metallicity": "st_met",
            "star_luminosity": "st_lum",
            "vmag": "sy_vmag",
            "kmag": "sy_kmag",
            "distance": "sy_dist",
            "star_spectral_type": "st_spectype",
        }

        provided = {
            "planet_radius": planet_radius_key,
            "planet_flux": planet_flux_key,
            "planet_equilibrium_temperature": planet_equilibrium_temperature_key,
            "semi_major_axis": semi_major_axis_key,
            "period": period_key,
            "msini": msini_key,
            "star_spectral_type": star_spectral_type_key,
            "star_radius": star_radius_key,
            "star_mass": star_mass_key,
            "star_effective_temperature": star_effective_temperature_key,
            "star_logg": star_logg_key,
            "star_metallicity": star_metallicity_key,
            "star_luminosity": star_luminosity_key,
            "vmag": vmag_key,
            "kmag": kmag_key,
            "distance": distance_key,
        }

        for key, value in provided.items():
            if value is not None:
                resolved[key] = value

        for key, value in override_keys.items():
            if value is not None:
                resolved[key] = value

        return resolved

        
    def _populate_output_quantity_columns(
        self,
        result: Table,
        params_q: dict[str, list[ParamQty]],
        resolved_cols: dict[str, str],
        upper_error_suffix: str,
        lower_error_suffix: str,
    ) -> None:
        for key in self.PARAM_NAMES_QUANTITIES:
            canonical_name, desc = self.PARAM_METADATA[key]
            col = resolved_cols[canonical_name]
            params = params_q[key]
            values = [p.val for p in params]
            masks = [p.mask or not np.isfinite(p.val) for p in params]
            result[col] = MaskedColumn(values,mask=masks,name=col,description=desc,)
            result[f"{col}_src"] = [p.src for p in params            ]
            result[f"{col}_{upper_error_suffix}"] = _mc([p.err1 for p in params], f"{col}_{upper_error_suffix}")
            result[f"{col}_{lower_error_suffix}"] = _mc([p.err2 for p in params], f"{col}_{lower_error_suffix}")


    def _populate_output_string_columns(
        self,
        result: Table,
        params_s: dict[str, list[ParamStr]],
        resolved_cols: dict[str, str],
        *,
        display_values: dict[str, list[str]] | None = None,
    ) -> None:
        display_values = display_values or {}

        for key in self.PARAM_NAMES_STRINGS:
            canonical_name, _ = self.PARAM_METADATA[key]
            col = resolved_cols[canonical_name]
            params = params_s[key]
            values = display_values.get(key)
            if values is None:
                values = [p.val for p in params]
            masks = [p.mask for p in params]
            result[col] = MaskedColumn(values,mask=masks,name=col)
            result[f"{col}_src"] = [p.src for p in params]

    def _preload_input_params(
        self,
        table: Table,
        params_q: dict[str, list[ParamQty]],
        params_s: dict[str, list[ParamStr]],
        resolved_cols: dict[str, str],
    ) -> None:
        n = len(table)

        for param_name in self.PARAM_NAMES_QUANTITIES:
            canonical_name, _ = self.PARAM_METADATA[param_name]
            col_name = resolved_cols[canonical_name]
            if col_name not in table.colnames:
                continue
            for i in range(n):
                params_q[param_name][i] = _qty_from_table(table, col_name, i)

        for param_name in self.PARAM_NAMES_STRINGS:
            canonical_name, _ = self.PARAM_METADATA[param_name]
            col_name = resolved_cols[canonical_name]
            if col_name not in table.colnames:
                continue
            for i in range(n):
                params_s[param_name][i] = _str_from_table(table, col_name, i)

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
    ) -> None:
        """Merge source-provided values into ParamQty and ParamStr containers.

        This method only performs priority-ordered source merging.
        Parameters
        ----------
        table : Table
            Input catalog table.
        params_q : dict[str, list[ParamQty]]
            Quantitative parameters to fill in-place.
        params_s : dict[str, list[ParamStr]]
            String parameters to fill in-place.
        upper_error_suffix : str
            Suffix for upper uncertainty fields.
        lower_error_suffix : str
            Suffix for lower uncertainty fields.
        input_starname_key : str, optional
            Column name used to identify the host star for source lookups.
        id_supplier : IdSupplierBase, optional
            ID supplier for cross-referencing identifiers.
        alternate_ids : Table, optional
            Pre-loaded alternate IDs table.
        """
        for i, row in enumerate(table):
            for source in self.sources:
                data = source.get(
                    row,
                    input_starname_key=input_starname_key,
                    id_supplier=id_supplier,
                    alternate_ids=alternate_ids,
                )
                for param in self.PARAM_NAMES_QUANTITIES:
                    source_key = self.SOURCE_KEY_ALIASES[param]
                    if source_key not in data: 
                        continue
                    param = params_q[param][i]
                    val = _to_float(data[source_key])
                    if not param.mask and np.isfinite(val):
                        continue
                    param.val = val
                    param.mask = False
                    param.err1 = _to_float(data.get(f"{source_key}_{upper_error_suffix}"))
                    param.err2 = _to_float(data.get(f"{source_key}_{lower_error_suffix}"))
                    param.src = source.source_name

                for param in self.PARAM_NAMES_STRINGS:
                    source_key = self.SOURCE_KEY_ALIASES[param]
                    if source_key not in data:
                        continue
                    param = params_s[param][i]
                    if not param.mask and not str(data[source_key] if data[source_key] is not None else ''):
                        continue
                    param.val = data[source_key]
                    param.mask = False
                    param.src = source.source_name    

    def enrich(
        self,
        table: Table,
        planet_radius_key: str | None = None,
        planet_flux_key: str | None = None,
        planet_equilibrium_temperature_key: str | None = None,
        semi_major_axis_key: str | None = None,
        period_key: str | None = None,
        msini_key: str | None = None,
        star_spectral_type_key: str | None = None,
        star_radius_key: str | None = None,
        star_mass_key: str | None = None,
        star_effective_temperature_key: str | None = None,
        star_logg_key: str | None = None,
        star_metallicity_key: str | None = None,
        star_luminosity_key: str | None = None,
        vmag_key: str | None = None,
        kmag_key: str | None = None,
        distance_key: str | None = None,
        upper_error_suffix: str = "err1",
        lower_error_suffix: str = "err2",
        input_starname_key: str | None = None,
        id_supplier: IdSupplierBase | None = None,
        alternate_ids: Table | None = None,
        disable_calculations: bool = False,
        **override_keys,
    ) -> Table:
        r"""Return a copy of table with enriched columns added or replaced.

        Data priority is:

        1. Values already present in the input table.
        2. Values supplied by configured ParamSource objects in priority order.
        3. Derived values, but only when disable_calculations is False.

        All physical and empirical derivations are performed through pure
        infer_* functions inside this method.

        Parameters
        ----------
        table : Table
            Input catalog table.
        planet_radius_key : str, optional
            Column name for planet radius.
        planet_flux_key : str, optional
            Column name for planet insolation flux.
        planet_equilibrium_temperature_key : str, optional
            Column name for planet equilibrium temperature.
        semi_major_axis_key : str, optional
            Column name for semi-major axis.
        period_key : str, optional
            Column name for orbital period.
        msini_key : str, optional
            Column name for minimum mass.
        star_spectral_type_key : str, optional
            Column name for stellar spectral type.
        star_radius_key : str, optional
            Column name for stellar radius.
        star_mass_key : str, optional
            Column name for stellar mass.
        star_effective_temperature_key : str, optional
            Column name for stellar effective temperature.
        star_logg_key : str, optional
            Column name for stellar log surface gravity.
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
            Suffix for upper uncertainty columns.
        lower_error_suffix : str, optional
            Suffix for lower uncertainty columns.
        input_starname_key : str, optional
            Column name used to identify host stars for lookups.
        id_supplier : IdSupplierBase, optional
            ID supplier for cross-referencing identifiers.
        alternate_ids : Table, optional
            Pre-loaded alternate IDs table.
        disable_calculations : bool, optional
            If True, only input/source merging is performed. No infer_* function is
            called.
        **override_keys
            Additional resolved-column overrides.

        Returns
        -------
        Table
            Enriched result table.
        """
        n = len(table)

        resolved_cols = self._resolved_columns(
            planet_radius_key=planet_radius_key,
            planet_flux_key=planet_flux_key,
            planet_equilibrium_temperature_key=planet_equilibrium_temperature_key,
            semi_major_axis_key=semi_major_axis_key,
            period_key=period_key,
            msini_key=msini_key,
            star_spectral_type_key=star_spectral_type_key,
            star_radius_key=star_radius_key,
            star_mass_key=star_mass_key,
            star_effective_temperature_key=star_effective_temperature_key,
            star_logg_key=star_logg_key,
            star_metallicity_key=star_metallicity_key,
            star_luminosity_key=star_luminosity_key,
            vmag_key=vmag_key,
            kmag_key=kmag_key,
            distance_key=distance_key,
            override_keys=override_keys,
        )

        params_q = {key: [ParamQty() for _ in range(n)] for key in self.PARAM_NAMES_QUANTITIES}
        params_s = {key: [ParamStr() for _ in range(n)] for key in self.PARAM_NAMES_STRINGS}

        # 1. Input table values have highest priority.
        self._preload_input_params(table, params_q, params_s, resolved_cols)

        # 2. Source values fill only missing values.
        self._merge_values(
            table,
            params_q,
            params_s,
            upper_error_suffix,
            lower_error_suffix,
            input_starname_key=input_starname_key,
            id_supplier=id_supplier,
            alternate_ids=alternate_ids,
        )

        result = table.copy()

        if disable_calculations:
            self._populate_output_quantity_columns(
                result,
                params_q,                resolved_cols,
                upper_error_suffix,
                lower_error_suffix,
            )
            self._populate_output_string_columns(
                result,
                params_s,
                resolved_cols,
            )

            return result
    
        # 3. Fundamental stellar inferences.
        r_lower_bound = [ParamQty() for _ in range(n)]
        r_upper_bound = [ParamQty() for _ in range(n)]
        semi_major_axis = [ParamQty() for _ in range(n)]

        displayed_spectypes: list[str] | None = None
        spectral_category: list[str] | None = None
        for i in range(n):
            params_q["st_teff"][i] = infer_star_teff(
                *(params_q[k][i] for k in ["st_teff", "st_rad", "st_lum"]),
                params_s["st_spectype"][i],
            )

            params_q["st_rad"][i] = infer_star_radius(
                *(params_q[k][i] for k in ["st_rad","st_mass", "st_logg", "st_teff", "st_lum", "st_met", "sy_kmag", "sy_dist"]),
                params_s["st_spectype"][i],
            )

            params_q["st_mass"][i] = infer_star_mass(
                *(params_q[k][i] for k in ["st_mass", "st_rad", "st_logg"]),
            )

            params_q["st_lum"][i] = infer_stellar_luminosity(
                *(params_q[k][i] for k in ["st_lum","st_rad",  "st_teff"]),
            )

            # 4. Planetary and orbital derived quantities.
            semi_major_axis_col = resolved_cols["semi_major_axis"]
            period_col = resolved_cols["period"]

            planet_radius = _qty_from_table(table,  resolved_cols["planet_radius"], i, src="input")
            msini = _qty_from_table(table, resolved_cols["msini"], i, src="input")

            r_lower_bound[i], r_upper_bound[i] = infer_msini_radius_bounds(planet_radius, msini, self.msini_sin_min)
            provided_a = _qty_from_table(table, semi_major_axis_col, i, src="provided")
            period = _qty_from_table(table, period_col, i, src="provided")
            semi_major_axis[i] = infer_semi_major_axis(provided_a, period, params_q["st_mass"][i],period_src=period_col)
            params_q["pl_insol"][i] = infer_planet_insolation(params_q["pl_insol"][i], params_q["st_lum"][i], semi_major_axis[i], params_q["pl_eqt"][i],)

            params_q["pl_eqt"][i] = infer_planet_equilibrium_temperature(params_q["pl_eqt"][i], params_q["pl_insol"][i], )

        # 5. Display spectral type and classify it.
        displayed_spectypes = [
            infer_spectral_type_display(params_s["st_spectype"][i], params_q["st_teff"][i]).val
            for i in range(n)
        ]
        spectral_category = [classify_spectral_type(s) for s in displayed_spectypes]

        # 6. Insert merged/inferred quantity columns.
        self._populate_output_quantity_columns(result,params_q,resolved_cols,upper_error_suffix,lower_error_suffix)
        self._populate_output_string_columns(result,params_s,resolved_cols,display_values={"st_spectype": displayed_spectypes,},)
        
        # 7. Insert extra derived columns.
        result["r_lower_bound"] = _mc( [q.val for q in r_lower_bound],"r_lower_bound","Min estimated planet radius from msini [R_earth]")
        result["r_upper_bound"] = _mc( [q.val for q in r_upper_bound],"r_upper_bound", "Max estimated planet radius from msini [R_earth]")
        result["a_src"] = [a.src for a in semi_major_axis]
        result["spectral_category"] = spectral_category
        return result
