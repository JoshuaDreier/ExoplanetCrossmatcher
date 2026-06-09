"""End-to-end test: Proxima Cen b is a temperate uncertain rocky planet.

Full pipeline:
  EMCCatalog + EMCIdSupplier → crossmatch TIC 388857263 (Proxima Centauri)
  → HpicStellarParamSource (HPIC stellar params from query row)
  → StellarParamMerger.enrich()
  → rocky_mask(use_interval=True) and temperate_mask() both True for Proxima Cen b

Uses only exo-mercat.csv (cached file, no network). The test is skipped
automatically if the file is absent from the working directory.
"""
import os

import numpy as np
import pytest
from astropy.table import Table

from crossmatching import Crossmatcher, EMCCatalog, EMCIdSupplier, StellarParamMerger
from crossmatching.enrichment import rocky_mask, temperate_mask
from crossmatching.param_sources.hpic import HpicStellarParamSource

_EMC_FILE = "exo-mercat.csv"
_HZ_LOWER  = 0.35   # S_Earth outer edge
_HZ_UPPER  = 1.77   # S_Earth inner edge
_ROCKY_LOWER = 0.5  # R_Earth
_ROCKY_UPPER = 1.5  # R_Earth

pytestmark = pytest.mark.skipif(
    not os.path.exists(_EMC_FILE),
    reason=f"{_EMC_FILE} not present — run from project root with cached catalog",
)


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

    # ── enrich ──────────────────────────────────────────────────────────────
    hpic_src = HpicStellarParamSource(result)
    hpic_src.load()
    return StellarParamMerger([hpic_src]).enrich(result)


# ── crossmatch found the planet ──────────────────────────────────────────────

def test_proxima_cen_b_is_found(proxima_enriched):
    names = [str(n) for n in proxima_enriched["exo-mercat_name"]]
    assert any("Proxima" in n and n.endswith(" b") for n in names), (
        f"Proxima Cen b not found in crossmatch; got: {names}"
    )


# ── no direct radius (no transit observed) ───────────────────────────────────

def test_proxima_cen_b_no_direct_radius(proxima_enriched):
    idx = _proxima_b_idx(proxima_enriched)
    assert np.ma.is_masked(proxima_enriched["r_earth"][idx])


# ── msini-based radius estimates ─────────────────────────────────────────────

def test_proxima_cen_b_r_earth_min_is_rocky(proxima_enriched):
    idx = _proxima_b_idx(proxima_enriched)
    rmin = float(proxima_enriched["r_earth_min"][idx])
    assert not np.isnan(rmin)
    assert _ROCKY_LOWER < rmin < _ROCKY_UPPER, f"r_earth_min={rmin:.3f} outside rocky range"


def test_proxima_cen_b_r_earth_max_is_rocky(proxima_enriched):
    idx = _proxima_b_idx(proxima_enriched)
    rmax = float(proxima_enriched["r_earth_max"][idx])
    assert not np.isnan(rmax)
    assert _ROCKY_LOWER < rmax < _ROCKY_UPPER, f"r_earth_max={rmax:.3f} outside rocky range"


# ── insolation flux ───────────────────────────────────────────────────────────

def test_proxima_cen_b_temperate(proxima_enriched):
    idx = _proxima_b_idx(proxima_enriched)
    flux = float(proxima_enriched["flux_rel"][idx])
    assert not np.isnan(flux)
    assert _HZ_LOWER < flux < _HZ_UPPER, f"flux_rel={flux:.3f} outside HZ range"


# ── combined classification ───────────────────────────────────────────────────

def test_proxima_cen_b_is_temperate_uncertain_rocky(proxima_enriched):
    out = proxima_enriched
    idx = _proxima_b_idx(out)

    is_temperate = temperate_mask(
        out["flux_rel"], out["flux_rel_err1"], out["flux_rel_err2"],
        lower=_HZ_LOWER, upper=_HZ_UPPER,
    )[idx]
    is_uncertain_rocky = rocky_mask(
        out["r_earth"], out["r_earth_min"], out["r_earth_max"],
        lower=_ROCKY_LOWER, upper=_ROCKY_UPPER, use_interval=True,
    )[idx]

    assert is_temperate, "Proxima Cen b should be classified as temperate"
    assert is_uncertain_rocky, "Proxima Cen b should be classified as uncertain rocky"
