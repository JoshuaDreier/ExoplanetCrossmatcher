"""Functional test: Proxima Cen b is temperate and uncertain rocky.

Uses canonical published parameters from Anglada-Escudé et al. 2016 (Nature 536, 437):
  msini  = 1.27 M_Earth
  a      = 0.0485 AU  (P = 11.186 d)
  Stellar: T_eff = 3050 K, R = 0.154 R_sun  (Boyajian et al. 2012)

These values are independent of the current EMC catalog state. The test exercises
the full enrichment pipeline (msini → r_lower_bound/max, pl_insol, classification)
without any network calls or file I/O.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from crossmatching.enrichment import (
    R_JUP_TO_EARTH,
    ParamFiller,
    mass_radius_chen_kipping,
    rocky_mask,
    temperate_mask,
)
from crossmatching.enrichment.param_sources.base import ParamSource


_MSINI_PROXIMA_B = 1.27 / u.M_jup.to(u.M_earth)  # 1.27 M_Earth in M_Jup

# HZ bounds used in this test (Kopparapu et al. 2013 conservative estimate)
_HZ_LOWER = 0.35   # S_Earth (outer edge)
_HZ_UPPER = 1.77   # S_Earth (inner edge)
_ROCKY_LOWER = 0.5  # R_Earth
_ROCKY_UPPER = 1.5  # R_Earth


class _FixedParamSource(ParamSource):
    """Toy source that returns pre-specified params for a given key."""

    key_col = "main_id"
    source_name = "test"

    def __init__(self, data: dict):
        self._lookup = data

    def download(self, key_list):
        raise NotImplementedError

    def _build_lookup(self, table):
        return {}


@pytest.fixture(scope="module")
def proxima_enriched():
    src = _FixedParamSource({
        "* Proxima Cen": {"teff": 3050.0, "rad": 0.154, "mass": 0.122},
    })

    table = Table({
        "exo-mercat_name": ["NAME Proxima Centauri b"],
        "nasa_name":       ["Proxima Cen b"],
        "main_id":         ["* Proxima Cen"],
        "toi_name":        [""],
        "epic_name":       [""],
        "r":               [np.nan],       # no transit radius
        "msini":           [_MSINI_PROXIMA_B],
        "a":               [0.0485],       # AU
        "p":               [11.186],       # days
    })

    merger = ParamFiller([src])
    return merger.enrich(table)


# ── radius from msini ─────────────────────────────────────────────────────────

def test_no_direct_radius(proxima_enriched):
    # r=NaN in input → not a valid transit radius; bounds columns are used instead
    assert not (float(proxima_enriched["r"][0]) > 0)


def test_r_lower_bound_in_rocky_range(proxima_enriched):
    rmin = float(proxima_enriched["r_lower_bound"][0])
    assert not np.isnan(rmin)
    assert _ROCKY_LOWER < rmin < _ROCKY_UPPER


def test_r_lower_bound_matches_chen_kipping(proxima_enriched):
    # r_min = mass_radius_chen_kipping(msini) at sin(i)=1 (edge-on)
    expected = mass_radius_chen_kipping(1.27)
    assert float(proxima_enriched["r_lower_bound"][0]) == pytest.approx(expected, rel=1e-4)


def test_r_upper_bound_matches_chen_kipping(proxima_enriched):
    # r_max = mass_radius_chen_kipping(msini / sin_min) with default sin_min=0.3
    expected = mass_radius_chen_kipping(1.27 / 0.3)
    assert float(proxima_enriched["r_upper_bound"][0]) == pytest.approx(expected, rel=1e-4)


# ── insolation flux ───────────────────────────────────────────────────────────

def test_pl_insol_in_temperate_zone(proxima_enriched):
    flux = float(proxima_enriched["pl_insol"][0])
    assert not np.isnan(flux)
    assert _HZ_LOWER < flux < _HZ_UPPER


def test_pl_insol_approx_value(proxima_enriched):
    # L ≈ 0.154² × (3050/5778)⁴ ≈ 0.00184 L_sun; F = L/a² ≈ 0.78 S_Earth
    assert float(proxima_enriched["pl_insol"][0]) == pytest.approx(0.78, rel=0.05)


# ── mask functions ────────────────────────────────────────────────────────────

def test_rocky_mask_use_interval(proxima_enriched):
    out = proxima_enriched
    assert not rocky_mask(out["r"] * R_JUP_TO_EARTH, out["r_lower_bound"], out["r_upper_bound"],
                          lower=_ROCKY_LOWER, upper=_ROCKY_UPPER)[0]
    assert rocky_mask(out["r"] * R_JUP_TO_EARTH, out["r_lower_bound"], out["r_upper_bound"],
                      lower=_ROCKY_LOWER, upper=_ROCKY_UPPER, use_interval=True)[0]


def test_temperate_mask(proxima_enriched):
    out = proxima_enriched
    assert temperate_mask(out["pl_insol"], out["pl_insol_err1"], out["pl_insol_err2"],
                          lower=_HZ_LOWER, upper=_HZ_UPPER)[0]


def test_proxima_cen_b_temperate_uncertain_rocky(proxima_enriched):
    out = proxima_enriched
    is_temperate = temperate_mask(
        out["pl_insol"], out["pl_insol_err1"], out["pl_insol_err2"],
        lower=_HZ_LOWER, upper=_HZ_UPPER,
    )[0]
    is_uncertain_rocky = rocky_mask(
        out["r"] * R_JUP_TO_EARTH, out["r_lower_bound"], out["r_upper_bound"],
        lower=_ROCKY_LOWER, upper=_ROCKY_UPPER, use_interval=True,
    )[0]
    assert is_temperate and is_uncertain_rocky
