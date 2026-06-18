from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
from crossmatching.enrichment.schema import (
    EXTRA_COLUMN_DEFAULTS,
    EXTRA_KEY_ALIASES,
    PARAM_KEY_ALIASES,
    PARAM_SPECS,
    PARAM_SPECS_BY_CANONICAL,
    QUANTITY_SPECS,
    STRING_SPECS,
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


@dataclass(frozen=True)
class ResolvedColumns:
    """Column names resolved from defaults plus caller overrides."""

    values: dict[str, str]

    def col(self, canonical_name: str) -> str:
        return self.values[canonical_name]

    def output_col(self, canonical_name: str) -> str:
        return self.col(canonical_name)

    def extra_col(self, name: str) -> str:
        return self.col(name)


@dataclass(frozen=True)
class ParamContext:
    """Convenient row-level access to canonical parameter containers."""

    params_q: dict[str, list[ParamQty]]
    params_s: dict[str, list[ParamStr]]
    idx: int

    def q(self, canonical_name: str) -> ParamQty:
        return self.params_q[PARAM_SPECS_BY_CANONICAL[canonical_name].id][self.idx]

    def set_q(self, canonical_name: str, value: ParamQty) -> None:
        self.params_q[PARAM_SPECS_BY_CANONICAL[canonical_name].id][self.idx] = value

    def s(self, canonical_name: str) -> ParamStr:
        return self.params_s[PARAM_SPECS_BY_CANONICAL[canonical_name].id][self.idx]


@dataclass
class PlanetaryInferenceState:
    table: Table
    resolved_cols: ResolvedColumns
    r_lower_bound: list[ParamQty]
    r_upper_bound: list[ParamQty]
    semi_major_axis: list[ParamQty]


class ParamFiller:
    """Enriches a catalog table with stellar and derived planetary parameters."""

    PARAM_SPECS = PARAM_SPECS
    PARAM_NAMES_QUANTITIES = [spec.id for spec in QUANTITY_SPECS]
    PARAM_NAMES_STRINGS = [spec.id for spec in STRING_SPECS]
    PARAM_NAMES = PARAM_NAMES_QUANTITIES + PARAM_NAMES_STRINGS
    PARAM_METADATA = {
        spec.id: (spec.canonical_name, spec.description) for spec in PARAM_SPECS
    }
    SOURCE_KEY_ALIASES = {spec.id: spec.source_key for spec in PARAM_SPECS}
    STELLAR_INFERENCE_STEPS = (
        "_infer_star_teff",
        "_infer_star_radius",
        "_infer_star_mass",
        "_infer_stellar_luminosity",
    )
    PLANETARY_INFERENCE_STEPS = (
        "_infer_msini_radius_bounds",
        "_infer_semi_major_axis",
        "_infer_planet_flux",
        "_infer_planet_equilibrium_temperature",
    )

    def __init__(self, sources: list[ParamSource], msini_sin_min: float = 0.3):
        self.sources = sources
        self.msini_sin_min = msini_sin_min

    def _q(self, params_q: dict[str, list[ParamQty]], canonical_name: str) -> list[ParamQty]:
        return params_q[PARAM_SPECS_BY_CANONICAL[canonical_name].id]

    def _s(self, params_s: dict[str, list[ParamStr]], canonical_name: str) -> list[ParamStr]:
        return params_s[PARAM_SPECS_BY_CANONICAL[canonical_name].id]

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
    ) -> ResolvedColumns:
        resolved = {
            **EXTRA_COLUMN_DEFAULTS,
            **{spec.canonical_name: spec.default_col for spec in PARAM_SPECS},
        }

        provided = {
            "planet_radius_key": planet_radius_key,
            "semi_major_axis_key": semi_major_axis_key,
            "period_key": period_key,
            "msini_key": msini_key,
            "planet_radius": planet_radius_key,
            "planet_flux": planet_flux_key,
            "planet_equilibrium_temperature": planet_equilibrium_temperature_key,
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

        for raw_key, value in {**provided, **override_keys}.items():
            if value is not None:
                self._set_resolved_column(resolved, raw_key, value)

        return ResolvedColumns(resolved)

    def _set_resolved_column(
        self,
        resolved: dict[str, str],
        raw_key: str,
        value: str,
    ) -> None:
        canonical = PARAM_KEY_ALIASES.get(raw_key) or EXTRA_KEY_ALIASES.get(raw_key) or raw_key
        if canonical in resolved:
            resolved[canonical] = value

    def _populate_output_quantity_columns(
        self,
        result: Table,
        params_q: dict[str, list[ParamQty]],
        resolved_cols: ResolvedColumns,
        upper_error_suffix: str,
        lower_error_suffix: str,
    ) -> None:
        for spec in QUANTITY_SPECS:
            col = resolved_cols.output_col(spec.canonical_name)
            params = params_q[spec.id]
            values = [p.val for p in params]
            masks = [p.mask or not np.isfinite(p.val) for p in params]
            result[col] = MaskedColumn(values, mask=masks, name=col, description=spec.description)
            result[f"{col}_src"] = [p.src for p in params]
            result[f"{col}_{upper_error_suffix}"] = _mc(
                [p.err1 for p in params],
                f"{col}_{upper_error_suffix}",
            )
            result[f"{col}_{lower_error_suffix}"] = _mc(
                [p.err2 for p in params],
                f"{col}_{lower_error_suffix}",
            )

    def _populate_output_string_columns(
        self,
        result: Table,
        params_s: dict[str, list[ParamStr]],
        resolved_cols: ResolvedColumns,
        *,
        display_values: dict[str, list[str]] | None = None,
    ) -> None:
        display_values = display_values or {}

        for spec in STRING_SPECS:
            col = resolved_cols.output_col(spec.canonical_name)
            params = params_s[spec.id]
            values = display_values.get(spec.id)
            if values is None:
                values = [p.val for p in params]
            masks = [p.mask for p in params]
            result[col] = MaskedColumn(values, mask=masks, name=col, description=spec.description)
            result[f"{col}_src"] = [p.src for p in params]

    def _preload_input_params(
        self,
        table: Table,
        params_q: dict[str, list[ParamQty]],
        params_s: dict[str, list[ParamStr]],
        resolved_cols: ResolvedColumns,
    ) -> None:
        n = len(table)

        for spec in QUANTITY_SPECS:
            col_name = resolved_cols.output_col(spec.canonical_name)
            if col_name not in table.colnames:
                continue
            for i in range(n):
                params_q[spec.id][i] = _qty_from_table(table, col_name, i)

        for spec in STRING_SPECS:
            col_name = resolved_cols.output_col(spec.canonical_name)
            if col_name not in table.colnames:
                continue
            for i in range(n):
                params_s[spec.id][i] = _str_from_table(table, col_name, i)

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
                for spec in QUANTITY_SPECS:
                    if spec.source_key not in data:
                        continue
                    param = params_q[spec.id][i]
                    val = _to_float(data[spec.source_key])
                    if not param.mask and np.isfinite(val):
                        continue
                    param.val = val
                    param.mask = False
                    param.err1 = _to_float(data.get(f"{spec.source_key}_{upper_error_suffix}"))
                    param.err2 = _to_float(data.get(f"{spec.source_key}_{lower_error_suffix}"))
                    param.src = source.source_name

                for spec in STRING_SPECS:
                    if spec.source_key not in data:
                        continue
                    param = params_s[spec.id][i]
                    value = data[spec.source_key]
                    if not param.mask and not str(value if value is not None else ""):
                        continue
                    param.val = value
                    param.mask = False
                    param.src = source.source_name

    def _apply_stellar_inferences(
        self,
        params_q: dict[str, list[ParamQty]],
        params_s: dict[str, list[ParamStr]],
        idx: int,
    ) -> None:
        ctx = ParamContext(params_q, params_s, idx)
        for step_name in self.STELLAR_INFERENCE_STEPS:
            getattr(self, step_name)(ctx)

    def _infer_star_teff(self, ctx: ParamContext) -> None:
        ctx.set_q(
            "star_effective_temperature",
            infer_star_teff(
                ctx.q("star_effective_temperature"),
                ctx.q("star_radius"),
                ctx.q("star_luminosity"),
                ctx.s("star_spectral_type"),
            ),
        )

    def _infer_star_radius(self, ctx: ParamContext) -> None:
        ctx.set_q(
            "star_radius",
            infer_star_radius(
                ctx.q("star_radius"),
                ctx.q("star_mass"),
                ctx.q("star_logg"),
                ctx.q("star_effective_temperature"),
                ctx.q("star_luminosity"),
                ctx.q("star_metallicity"),
                ctx.q("kmag"),
                ctx.q("distance"),
                ctx.s("star_spectral_type"),
            ),
        )

    def _infer_star_mass(self, ctx: ParamContext) -> None:
        ctx.set_q(
            "star_mass",
            infer_star_mass(
                ctx.q("star_mass"),
                ctx.q("star_radius"),
                ctx.q("star_logg"),
            ),
        )

    def _infer_stellar_luminosity(self, ctx: ParamContext) -> None:
        ctx.set_q(
            "star_luminosity",
            infer_stellar_luminosity(
                ctx.q("star_luminosity"),
                ctx.q("star_radius"),
                ctx.q("star_effective_temperature"),
            ),
        )

    def _apply_planetary_inferences(
        self,
        table: Table,
        params_q: dict[str, list[ParamQty]],
        resolved_cols: ResolvedColumns,
        idx: int,
        r_lower_bound: list[ParamQty],
        r_upper_bound: list[ParamQty],
        semi_major_axis: list[ParamQty],
    ) -> None:
        ctx = ParamContext(params_q, {}, idx)
        state = PlanetaryInferenceState(
            table=table,
            resolved_cols=resolved_cols,
            r_lower_bound=r_lower_bound,
            r_upper_bound=r_upper_bound,
            semi_major_axis=semi_major_axis,
        )
        for step_name in self.PLANETARY_INFERENCE_STEPS:
            getattr(self, step_name)(ctx, state)

    def _infer_msini_radius_bounds(self, ctx: ParamContext, state: PlanetaryInferenceState) -> None:
        table = state.table
        resolved_cols = state.resolved_cols
        planet_radius = _qty_from_table(
            table,
            resolved_cols.extra_col("planet_radius"),
            ctx.idx,
            src="input",
        )
        msini = _qty_from_table(table, resolved_cols.extra_col("msini"), ctx.idx, src="input")

        state.r_lower_bound[ctx.idx], state.r_upper_bound[ctx.idx] = infer_msini_radius_bounds(
            planet_radius,
            msini,
            self.msini_sin_min,
        )

    def _infer_semi_major_axis(self, ctx: ParamContext, state: PlanetaryInferenceState) -> None:
        table = state.table
        resolved_cols = state.resolved_cols
        provided_a = _qty_from_table(
            table,
            resolved_cols.extra_col("semi_major_axis"),
            ctx.idx,
            src="provided",
        )
        period = _qty_from_table(table, resolved_cols.extra_col("period"), ctx.idx, src="provided")
        state.semi_major_axis[ctx.idx] = infer_semi_major_axis(
            provided_a,
            period,
            ctx.q("star_mass"),
            period_src=resolved_cols.extra_col("period"),
        )

    def _infer_planet_flux(self, ctx: ParamContext, state: PlanetaryInferenceState) -> None:
        ctx.set_q(
            "planet_flux",
            infer_planet_insolation(
                ctx.q("planet_flux"),
                ctx.q("star_luminosity"),
                state.semi_major_axis[ctx.idx],
                ctx.q("planet_equilibrium_temperature"),
            ),
        )

    def _infer_planet_equilibrium_temperature(
        self,
        ctx: ParamContext,
        _state: PlanetaryInferenceState,
    ) -> None:
        ctx.set_q(
            "planet_equilibrium_temperature",
            infer_planet_equilibrium_temperature(
                ctx.q("planet_equilibrium_temperature"),
                ctx.q("planet_flux"),
            ),
        )

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
            **override_keys,
        )

        params_q = {spec.id: [ParamQty() for _ in range(n)] for spec in QUANTITY_SPECS}
        params_s = {spec.id: [ParamStr() for _ in range(n)] for spec in STRING_SPECS}

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
                params_q,
                resolved_cols,
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

        for i in range(n):
            self._apply_stellar_inferences(params_q, params_s, i)
            self._apply_planetary_inferences(
                table,
                params_q,
                resolved_cols,
                i,
                r_lower_bound,
                r_upper_bound,
                semi_major_axis,
            )

        # 5. Display spectral type and classify it.
        spectype = self._s(params_s, "star_spectral_type")
        teff = self._q(params_q, "star_effective_temperature")
        displayed_spectypes = [
            infer_spectral_type_display(spectype[i], teff[i]).val
            for i in range(n)
        ]
        spectral_category = [classify_spectral_type(s) for s in displayed_spectypes]

        # 6. Insert merged/inferred quantity columns.
        self._populate_output_quantity_columns(
            result,
            params_q,
            resolved_cols,
            upper_error_suffix,
            lower_error_suffix,
        )
        self._populate_output_string_columns(
            result,
            params_s,
            resolved_cols,
            display_values={
                PARAM_SPECS_BY_CANONICAL["star_spectral_type"].id: displayed_spectypes,
            },
        )

        # 7. Insert extra derived columns.
        result["r_lower_bound"] = _mc(
            [q.val for q in r_lower_bound],
            "r_lower_bound",
            "Min estimated planet radius from msini [R_earth]",
        )
        result["r_upper_bound"] = _mc(
            [q.val for q in r_upper_bound],
            "r_upper_bound",
            "Max estimated planet radius from msini [R_earth]",
        )
        result["a_src"] = [a.src for a in semi_major_axis]
        result["spectral_category"] = spectral_category
        return result
