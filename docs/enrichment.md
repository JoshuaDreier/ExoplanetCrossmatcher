# Enrichment

> [!warning]
> Enrichment is meant for broad candidate filtering and sanity checks. Many values are merged from heterogeneous catalogues or estimated from approximate relations; do not treat the output as a publication-ready per-planet parameter table without validating the underlying sources.

`ParamFiller.enrich()` returns a copy of an Astropy `Table` with stellar, orbital, and derived planet columns filled in. It is usually run after crossmatching, but it can also enrich a loaded catalogue table directly.

The current enrichment API is centered on:

- `ParamFiller` in `crossmatching.enrichment.merger`
- `ParamSource` classes in `crossmatching.enrichment.param_sources`
- `rocky_mask()` and `temperate_mask()` in `crossmatching.enrichment.masks`

## Basic Example

For a table produced by `EMCCatalog`, use the catalogue's built-in key mapping so `ParamFiller` knows which Exo-MerCat columns contain radius, period, semi-major axis, and `msini`.

```python
from crossmatching import EMCCatalog, EMCIdSupplier, Crossmatcher, ParamFiller
from crossmatching.enrichment import NeaParamSource, SimbadParamSource, HpicParamSource

# Assume this is the result of cme.combined_crossmatch(...)
emc_result = cme.combined_crossmatch(input_table, input_starname_key="star_name")

nea_src = NeaParamSource()
nea_src.load(from_file="./input/pscomppars.txt", format="ascii")

simbad_src = SimbadParamSource()
simbad_src.load(from_file="./input/simbad_params.txt", format="ascii")

hpic_src = HpicParamSource(emc_result)
hpic_src.load()

filler = ParamFiller([hpic_src, nea_src, simbad_src])

enriched = filler.enrich(
    emc_result,
    input_starname_key="star_name",
    id_supplier=cme.id_supplier,
    alternate_ids=cme.alternate_ids,
    **EMCCatalog.ENRICH_KEYS,
)

enriched[
    "exo-mercat_name",
    "st_teff", "st_teff_src",
    "st_rad", "st_rad_src",
    "pl_insol", "pl_insol_src",
    "r_lower_bound", "r_upper_bound",
    "spectral_category",
]
```

For a NASA Exoplanet Archive table, use `NEACatalog.ENRICH_KEYS` instead:

```python
from crossmatching import NEACatalog

enriched_nea = filler.enrich(
    cm.catalog_table,
    input_starname_key="pl_name",
    id_supplier=cm.id_supplier,
    alternate_ids=cm.alternate_ids,
    **NEACatalog.ENRICH_KEYS,
)
```

If you only want to enrich a small table from one source, the minimum version is:

```python
from crossmatching import ParamFiller
from crossmatching.enrichment import NeaParamSource

nea_src = NeaParamSource()
nea_src.load(from_file="./input/pscomppars.txt", format="ascii")

enriched = ParamFiller([nea_src]).enrich(table, **NEACatalog.ENRICH_KEYS)
```

## Priority Rules

Values are selected row by row in this order:

1. Existing non-masked values already present in the input table.
2. Values from the configured `ParamSource` objects, in the order passed to `ParamFiller`.
3. Derived values from `crossmatching.enrichment.inference`, unless `disable_calculations=True`.

This means the source order is a scientific choice. A common order for HPIC plus Exo-MerCat work is:

```python
filler = ParamFiller([
    hpic_src,    # values already tied to the crossmatched HPIC rows
    nea_src,     # NASA Exoplanet Archive composite parameters
    eu_src,      # exoplanet.eu
    epic_src,    # K2 candidates and planets
    toi_src,     # TESS Objects of Interest
    simbad_src,  # broad fallback for stellar data
])
```

Every enriched quantity gets a provenance column named `<column>_src`, for example `st_rad_src` or `pl_insol_src`. Direct source values use the source label such as `hpic`, `nea`, `epic`, `toi`, `eu`, or `simbad`. Derived values use strings such as `derived(rad:nea teff:hpic)` or `kepler(mass:nea period:p)`.

## Parameter Sources

All parameter sources follow the same pattern:

```python
src = NeaParamSource()
src.load(from_file="cached_source_file.txt", format="ascii")
```

If `from_file` is omitted, the source downloads from its TAP service. During development, prefer caching with `save_raw(...)` or loading a local file.

| Source              | Lookup key in enriched row | Typical fields                                                                                                        |
| ------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `HpicParamSource`   | `exo-mercat_name`          | `st_teff`, `st_rad`, `sy_vmag`, `st_spectype`                                                                         |
| `NeaParamSource`    | `nasa_name`                | stellar radius, mass, Teff, logg, metallicity, luminosity, magnitudes, distance, planet flux, equilibrium temperature |
| `EpicParamSource`   | `epic_name`                | K2/EPIC stellar and planet parameters                                                                                 |
| `ToiParamSource`    | `toi_name`                 | TOI Teff, radius, distance, flux, logg, equilibrium temperature                                                       |
| `EuParamSource`     | `name`                     | exoplanet.eu stellar parameters (no errors) and equilibrium temperature                                               |
| `SimbadParamSource` | `main_id`                  | spectral type, Teff, logg, metallicity, V/K magnitudes, distance (from parallax)                                      |

Source lookup first tries the source's direct key column. If that fails and you passed `input_starname_key`, `id_supplier`, and `alternate_ids`, it can try alternate identifiers from the ID supplier.

## Column Keys

`enrich()` has defaults such as `st_rad`, `st_teff`, `pl_insol`, `pl_eqt`, `pl_a`, `period`, and `msini`, but real catalogues use different names. Prefer the built-in mappings:

```python
enriched = filler.enrich(result, **EMCCatalog.ENRICH_KEYS)
enriched = filler.enrich(result, **NEACatalog.ENRICH_KEYS)
```

Important keys include:

| Logical value           | Keyword                              | Default    | EMC       | NEA          |
| ----------------------- | ------------------------------------ | ---------- | --------- | ------------ |
| Planet radius           | `planet_radius_key`                  | `pl_rad`   | `r`       | `pl_radj`    |
| Planet mass             | `planet_mass_key`                    | `pl_mass`  | `mass`    | `pl_massj`   |
| Planet flux             | `planet_flux_key`                    | `pl_insol` | N/A       | `pl_insol`   |
| Equilibrium temperature | `planet_equilibrium_temperature_key` | `pl_eqt`   | N/A       | `pl_eqt`     |
| Semi-major axis         | `semi_major_axis_key`                | `pl_a`     | `a`       | `pl_orbsmax` |
| Period                  | `period_key`                         | `period`   | `p`       | `pl_orbper`  |
| Minimum mass            | `msini_key`                          | `msini`    | `msini`   | `pl_msinij`  |
| Stellar radius          | `star_radius_key`                    | `st_rad`   | `st_rad`  | `st_rad`     |
| Stellar Teff            | `star_effective_temperature_key`     | `st_teff`  | `st_teff` | `st_teff`    |


The resolved key controls the output name. For example, `planet_radius_key="r"` produces `r_lower_bound` and `r_upper_bound`; `semi_major_axis_key="a"` writes the enriched semi-major axis back to `a` plus `a_src`, `a_err1`, and `a_err2`.

## Output Columns

For each enriched quantity column, `ParamFiller` writes:

- `<column>`: selected or inferred value as a masked column
- `<column>_src`: provenance string
- `<column>_err1`: upper 1-sigma uncertainty, positive magnitude
- `<column>_err2`: lower 1-sigma uncertainty, positive magnitude

(`err1`, `err2` suffixes can be changed with keyword arguments, they are `_max` , `_min`  for Exo-MerCat)
The standard quantity outputs are:

| Default column | Meaning                       | Unit convention    |
| -------------- | ----------------------------- | ------------------ |
| `st_rad`       | Stellar radius                | solar radii        |
| `st_mass`      | Stellar mass                  | solar masses       |
| `st_teff`      | Stellar effective temperature | K                  |
| `st_logg`      | Stellar surface gravity       | log10(cm/s^2)      |
| `st_met`       | Stellar metallicity           | dex                |
| `st_lum`       | Stellar luminosity            | solar luminosities |
| `sy_vmag`      | V magnitude                   | mag                |
| `sy_kmag`      | K magnitude                   | mag                |
| `sy_dist`      | Distance                      | pc                 |
| `pl_insol`     | Planet insolation             | Earth flux         |
| `pl_eqt`       | Equilibrium temperature       | K                  |
| `pl_a`         | Semi-major axis               | AU                 |
| `st_spectype`  | Display spectral type         | string             |
| `period`       | Orbital Period                | days               |
| `pl_rad`       | Planet radius                 | Jupiter radii      |
| `pl_mass`      | Planet mass                   | Jupiter masses     |

When calculations are enabled, two extra planet-radius bound columns are added using the resolved planet-radius key:
- `<planet_radius_key>_lower_bound`
- `<planet_radius_key>_upper_bound`
With `EMCCatalog.ENRICH_KEYS`, these are `r_lower_bound` and `r_upper_bound`. They are only filled when direct planet radius is absent and `msini` is available.

`spectral_category` is also added with one of `Sun-like`, `Low-luminosity`, `Very-low-luminosity`, or `Other`.

## Derived Values

When `disable_calculations=False`, missing values can be inferred:

| Quantity              | Inference path                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `st_teff`             | Stefan-Boltzmann from luminosity and radius, then spectral type fallback                                             |
| `st_rad`              | mass plus logg, luminosity plus Teff, Mann 2015 K-band/Teff relations, Torres 2010, then main-sequence Teff fallback |
| `st_mass`             | radius plus logg                                                                                                     |
| `st_lum`              | `R^2 * (Teff / 5778 K)^4`                                                                                            |
| semi-major axis       | Kepler's third law from period and stellar mass                                                                      |
| `pl_insol`            | luminosity divided by semi-major axis squared, then equilibrium-temperature fallback                                 |
| `pl_eqt`              | `254.793 * pl_insol**0.25`                                                                                           |
| radius bounds         | Chen-Kipping mass-radius relation from `msini`                                                                       |
| `st_spectype` display | source spectral type, falling back to Teff-derived approximate type                                                  |

The detailed formulas, validity notes, provenance strings, and literature references are documented in [Derived Parameter Inference](derived-parameter-inference.md).

To perform source merging only, disable calculations:

```python
merged_only = filler.enrich(
    result,
    disable_calculations=True,
    **EMCCatalog.ENRICH_KEYS,
)
```

In this mode no `infer_*` functions are called, and derived-only columns such as `r_lower_bound`, `r_upper_bound`, and `spectral_category` are not added.

## Candidate Masks

Use `rocky_mask()` and `temperate_mask()` for broad filtering:

```python
from astropy import units as u
from crossmatching import rocky_mask, temperate_mask

radius_earth = enriched["r"] * u.R_jup.to(u.R_earth)

is_rocky = rocky_mask(
    radius_earth,
    enriched["r_lower_bound"],
    enriched["r_upper_bound"],
    lower=0.5,
    upper=1.5,
)

could_be_rocky = rocky_mask(
    radius_earth,
    enriched["r_lower_bound"],
    enriched["r_upper_bound"],
    lower=0.5,
    upper=1.5,
    use_interval=True,
)

is_temperate = temperate_mask(
    enriched["pl_insol"],
    enriched["pl_insolerr1"],
    enriched["pl_insolerr2"],
    lower=0.35,
    upper=1.77,
)

candidates = enriched[could_be_rocky & is_temperate]
```

`use_interval=True` includes uncertain rows whose interval overlaps the target range. For `rocky_mask`, that means rows with no direct radius can be included if their `msini`-derived radius bounds overlap the rocky range.

## Important Notes
- `err1` and `err2` are stored as positive magnitudes. Add `err1` for the upper bound and subtract `err2` for the lower bound.
- Source order matters. Once a non-masked value is found, later sources do not replace it.
- Existing values in the input table have highest priority, even over the first `ParamSource`.
- The planet-radius key is not unit-converted by `ParamFiller`. For Exo-MerCat `r`, convert Jupiter radii to Earth radii before using `rocky_mask()` on direct radii.
- `msini` is treated as Jupiter masses in the radius-bound inference.
- NEA-style source luminosity values are converted from log10 luminosity to linear solar luminosities inside the source lookup.
- The derived radius bounds are intentionally broad. The upper bound uses `msini_sin_min=0.3` by default; override it with `ParamFiller(sources, msini_sin_min=...)`.
