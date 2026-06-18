from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ParamKind = Literal["quantity", "string"]


@dataclass(frozen=True)
class ParamSpec:
    """Describes one mergeable/enrichable output parameter."""

    id: str
    canonical_name: str
    default_col: str
    source_key: str
    kind: ParamKind
    description: str = ""
    key_aliases: tuple[str, ...] = ()


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec(
        id="st_rad",
        canonical_name="star_radius",
        default_col="st_rad",
        source_key="rad",
        kind="quantity",
        description="Stellar radius [R_sun]",
        key_aliases=("stellar_radius_key",),
    ),
    ParamSpec(
        id="st_mass",
        canonical_name="star_mass",
        default_col="st_mass",
        source_key="mass",
        kind="quantity",
        description="Stellar mass [M_sun]",
        key_aliases=("stellar_mass_key",),
    ),
    ParamSpec(
        id="st_teff",
        canonical_name="star_effective_temperature",
        default_col="st_teff",
        source_key="teff",
        kind="quantity",
        description="Stellar effective temperature [K]",
        key_aliases=("stellar_teff_key", "stellar_effective_temperature_key"),
    ),
    ParamSpec(
        id="st_logg",
        canonical_name="star_logg",
        default_col="st_logg",
        source_key="logg",
        kind="quantity",
        description="Stellar log surface gravity [log(cm/s^2)]",
        key_aliases=("stellar_logg_key",),
    ),
    ParamSpec(
        id="st_met",
        canonical_name="star_metallicity",
        default_col="st_met",
        source_key="met",
        kind="quantity",
        description="Stellar metallicity [Fe/H]",
        key_aliases=("stellar_metallicity_key",),
    ),
    ParamSpec(
        id="st_lum",
        canonical_name="star_luminosity",
        default_col="st_lum",
        source_key="lum",
        kind="quantity",
        description="Stellar luminosity [L_sun]",
        key_aliases=("stellar_luminosity_key",),
    ),
    ParamSpec(
        id="sy_vmag",
        canonical_name="vmag",
        default_col="sy_vmag",
        source_key="vmag",
        kind="quantity",
        description="Visual band magnitude [mag]",
    ),
    ParamSpec(
        id="sy_kmag",
        canonical_name="kmag",
        default_col="sy_kmag",
        source_key="kmag",
        kind="quantity",
        description="2MASS K-band magnitude [mag]",
    ),
    ParamSpec(
        id="sy_dist",
        canonical_name="distance",
        default_col="sy_dist",
        source_key="dist",
        kind="quantity",
        description="Distance [pc]",
    ),
    ParamSpec(
        id="pl_insol",
        canonical_name="planet_flux",
        default_col="pl_insol",
        source_key="insol",
        kind="quantity",
        description="Planet insolation flux [S_earth]",
        key_aliases=("flux_rel_key", "planet_insolation_key"),
    ),
    ParamSpec(
        id="pl_eqt",
        canonical_name="planet_equilibrium_temperature",
        default_col="pl_eqt",
        source_key="pl_eqt",
        kind="quantity",
        description="Planet equilibrium temperature [K]",
    ),
    ParamSpec(
        id="st_spectype",
        canonical_name="star_spectral_type",
        default_col="st_spectype",
        source_key="spec",
        kind="string",
        description="Stellar spectral type",
        key_aliases=("stellar_spectral_type_key",),
    ),
)

QUANTITY_SPECS = tuple(spec for spec in PARAM_SPECS if spec.kind == "quantity")
STRING_SPECS = tuple(spec for spec in PARAM_SPECS if spec.kind == "string")
PARAM_SPECS_BY_ID = {spec.id: spec for spec in PARAM_SPECS}
PARAM_SPECS_BY_CANONICAL = {spec.canonical_name: spec for spec in PARAM_SPECS}

PARAM_KEY_ALIASES = {
    f"{spec.canonical_name}_key": spec.canonical_name
    for spec in PARAM_SPECS
}
for spec in PARAM_SPECS:
    for alias in spec.key_aliases:
        PARAM_KEY_ALIASES[alias] = spec.canonical_name

EXTRA_COLUMN_DEFAULTS = {
    "planet_radius": "pl_rad",
    "semi_major_axis": "semi_major_axis",
    "period": "period",
    "msini": "msini",
}
EXTRA_KEY_ALIASES = {
    "planet_radius_key": "planet_radius",
    "semi_major_axis_key": "semi_major_axis",
    "period_key": "period",
    "msini_key": "msini",
}

