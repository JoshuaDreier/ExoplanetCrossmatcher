"""End-to-end test: Proxima Cen b is a temperate uncertain rocky planet.

Full pipeline:
  EMCCatalog + EMCIdSupplier → crossmatch TIC 388857263 (Proxima Centauri)
  → HpicParamSource (HPIC stellar params from query row)
  → ParamFiller.enrich()[0]
  → rocky_mask(use_interval=True) and temperate_mask() both True for Proxima Cen b

Uses only ./input/exo-mercat.csv (cached file, no network). The test is skipped
automatically if the file is absent from the working directory.
"""
import os, glob

import numpy as np
import pytest
from astropy.table import Table
import astropy.units as u

from crossmatching import Crossmatcher, EMCCatalog, EMCIdSupplier, ParamFiller
from crossmatching.enrichment import rocky_mask, temperate_mask
from crossmatching.enrichment import (
    HpicParamSource, NeaParamSource, SimbadParamSource,
    EpicParamSource, ToiParamSource, EuParamSource
)

_EMC_FILE   = "tests/data/exo-mercat2026-06-08.csv"
_HZ_LOWER  = 0.35   # S_Earth outer edge
_HZ_UPPER  = 1.77   # S_Earth inner edge
_ROCKY_LOWER = 0.5  # R_Earth
_ROCKY_UPPER = 1.5  # R_Earth


def _proxima_b_idx(table):
    names = list(table["exo-mercat_name"])
    for i, n in enumerate(names):
        if "Proxima" in str(n) and str(n).endswith(" b"):
            return i
    raise AssertionError(
        f"'NAME Proxima Centauri b' not found in enriched table; got: {names}"
    )


@pytest.fixture(scope="module")
def proxima_enriched():
    # ── crossmatch ──────────────────────────────────────────────────────────
    xm = Crossmatcher(EMCCatalog(), EMCIdSupplier())
    xm.load_catalog(from_file=_EMC_FILE, format="csv")
    xm.load_alternate_ids(["TIC 388857263"], from_file=_EMC_FILE)

    # HPIC query row — TIC 388857263 = Proxima Centauri (confirmed in HPIC file)
    # Stellar params from the HPIC catalog: st_teff=2979 K, st_rad=0.1465 R_sun
    query = Table({
        "star_name":   ["TIC 388857263"],
        "ra":          [217.429],
        "dec":         [-62.6795],
        "st_teff":     [2979.0],
        "st_rad":      [0.1465],
        "st_mass":     [0.1262],
        "sy_vmag":     [np.nan],
        "sy_dist":     [1.302],
        "st_spectype": [""],
    })

    result = xm.combined_crossmatch(query, "star_name")

    nea_src = NeaParamSource()
    nea_src.load(from_file='././input/pscomppars.txt', format='ascii')
    print(f'NEA planets loaded:     {len(nea_src._lookup):,}')
    print(f'With insol:             {sum(1 for v in nea_src._lookup.values() if "insol" in v):,}')
    print(f'With teff + rad:        {sum(1 for v in nea_src._lookup.values() if "teff" in v and "rad" in v):,}')

    eu_src = EuParamSource()
    eu_path = sorted(glob.glob('../Exo-MerCat/InputSources/eu_init*.csv'))[-1]
    eu_src.load(from_file=eu_path, format="ascii.csv")
    print(f"\nEU planets loaded: {len(eu_src._lookup):,}")

    epic_src = EpicParamSource()
    epic_path = sorted(glob.glob('../Exo-MerCat/InputSources/epic_init*.csv'))[-1]
    epic_src.load(from_file=epic_path, format='ascii.csv')
    print(f'\nEPIC planets loaded:    {len(epic_src._lookup):,}')
    print(f'With insol:             {sum(1 for v in epic_src._lookup.values() if "insol" in v):,}')
    print(f'With teff + rad:        {sum(1 for v in epic_src._lookup.values() if "teff" in v and "rad" in v):,}')

    toi_src = ToiParamSource()
    toi_path = sorted(glob.glob('../Exo-MerCat/InputSources/toi_init*.csv'))[-1]
    toi_src.load(from_file=toi_path, format='ascii.csv')
    print(f'\nTOI entries loaded:     {len(toi_src._lookup):,}')
    print(f'With insol:             {sum(1 for v in toi_src._lookup.values() if "insol" in v):,}')
    print(f'With teff + rad:        {sum(1 for v in toi_src._lookup.values() if "teff" in v and "rad" in v):,}')

    simbad_src = SimbadParamSource()
    simbad_src.load(from_file='./input/simbad_params.txt')
    print(f'SIMBAD matches: {len(simbad_src._lookup):,}')

    merger = ParamFiller([nea_src, eu_src, toi_src, toi_src, simbad_src])
    return merger.enrich(result, **EMCCatalog.ENRICH_KEYS)[0]


# ── crossmatch found the planet ──────────────────────────────────────────────

def test_proxima_cen_b_is_found(proxima_enriched):
    names = [str(n) for n in proxima_enriched["exo-mercat_name"]]
    assert any("Proxima" in n and n.endswith(" b") for n in names), (
        f"Proxima Cen b not found in crossmatch; got: {names}"
    )


# ── no direct radius (no transit observed) ───────────────────────────────────

def test_proxima_cen_b_no_direct_radius(proxima_enriched):
    idx = _proxima_b_idx(proxima_enriched)
    assert proxima_enriched["r"].mask[idx]  # no transit radius


# ── msini-based radius estimates ─────────────────────────────────────────────

def test_proxima_cen_b_r_lower_bound_is_rocky(proxima_enriched):
    idx = _proxima_b_idx(proxima_enriched)
    rmin = float(proxima_enriched["r_lower_bound"][idx])
    assert not np.isnan(rmin)
    assert _ROCKY_LOWER < rmin < _ROCKY_UPPER, f"r_lower_bound={rmin:.3f} outside rocky range"

# ── insolation flux ───────────────────────────────────────────────────────────

def test_proxima_cen_b_temperate(proxima_enriched):
    idx = _proxima_b_idx(proxima_enriched)
    flux = float(proxima_enriched["pl_insol"][idx])
    assert not np.isnan(flux)
    assert _HZ_LOWER < flux < _HZ_UPPER, f"pl_insol={flux:.3f} outside HZ range"


# ── combined classification ───────────────────────────────────────────────────

def test_proxima_cen_b_is_temperate_uncertain_rocky(proxima_enriched):
    out = proxima_enriched
    idx = _proxima_b_idx(out)

    out['r_earth'] = out['r'] * u.R_jup.to(u.R_earth)
    out['r_earth_max'] = out['r_max'] * u.R_jup.to(u.R_earth)
    out['r_earth_min'] = out['r_min'] * u.R_jup.to(u.R_earth)
    out['r_earth_upper_bound'] = out['r_upper_bound'] * u.R_jup.to(u.R_earth)
    out['r_earth_lower_bound'] = out['r_lower_bound'] * u.R_jup.to(u.R_earth)

    is_temperate = temperate_mask(
        *([i] for i in out["pl_insol", "pl_insol_max", "pl_insol_min"][idx]),
        lower=_HZ_LOWER, upper=_HZ_UPPER,
    )
    is_uncertain_rocky = rocky_mask(
        *([i] for i in out["r", "r_earth_min", "r_earth_max", "r_earth_lower_bound", "r_earth_upper_bound"][idx]),
        lower=_ROCKY_LOWER, upper=_ROCKY_UPPER, use_interval=True,
    )

    assert is_temperate, "Proxima Cen b should be classified as temperate"
    assert is_uncertain_rocky, "Proxima Cen b should be classified as uncertain rocky"
