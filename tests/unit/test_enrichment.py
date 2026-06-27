import numpy as np
import pytest
from astropy.table import MaskedColumn

from crossmatching.enrichment import (
    classify_spectral_type,
    mass_radius_chen_kipping,
    ms_radius_from_teff,
    rocky_mask,
    spectype_display,
    temperate_mask,
    teff_to_spectype,
    _mann_teff_radius,
    _mann_mks_radius,
    _torres_radius,
)


# --- classify_spectral_type: boundary and regex-branch cases ---

@pytest.mark.parametrize("sptype,expected", [
    ("K5V",  "Sun-like"),           # K5 boundary — inclusive (subtype <= 5)
    ("K6V",  "Low-luminosity"),     # K6 boundary — first value excluded from Sun-like
    ("M2.5V","Low-luminosity"),     # float subtype parsed correctly
    ("M3V",  "Very-low-luminosity"),# M3 boundary (subtype < 3 → low, >= 3 → very-low)
    ("dM2",  "Low-luminosity"),     # leading 'd' dwarf prefix stripped before letter match
    ("dK5",  "Sun-like"),           # 'd' prefix combined with K5 boundary
    ("K2IV", "Other"),              # subgiant: re.search catches IV before main branch
    ("DA2",  "Other"),              # white dwarf: ^D[A-Z0-9] pattern
    ("null", "Other"),              # sentinel string — no letter regex match
    ("",     "Other"),              # empty string — no match
])
def test_classify_spectral_type(sptype, expected):
    assert classify_spectral_type(sptype) == expected


# --- teff_to_spectype ---

def test_teff_to_spectype_m_dwarf():
    assert teff_to_spectype(3500.0) == "~M2"


def test_teff_to_spectype_zero():
    assert teff_to_spectype(0.0) == ""


def test_teff_to_spectype_negative():
    assert teff_to_spectype(-100.0) == ""


# --- ms_radius_from_teff ---

def test_ms_radius_zero_guard():
    assert ms_radius_from_teff(0.0) == 0.0


def test_ms_radius_clamps_to_minimum():
    # Very low teff → (100/5778)^1.8 ≪ 0.05 → clamped to 0.05
    assert ms_radius_from_teff(100.0) == pytest.approx(0.05)


# --- spectype_display ---

def test_spectype_display_null_falls_back_to_teff():
    result = spectype_display("null", 3500.0)
    assert result == "~M2"


def test_spectype_display_zero_teff_and_null_spec_returns_empty():
    assert spectype_display("null", 0.0) == ""


# --- temperate_mask ---

def _flux(vals, mask=None):
    """Build a MaskedColumn of insolation values."""
    if mask is None:
        mask = [False] * len(vals)
    return MaskedColumn(vals, mask=mask, name='pl_insol')


def test_temperate_mask_central_in_range():
    flux = _flux([0.2, 1.0, 2.0])
    result = temperate_mask(flux, None, None, lower=0.25, upper=1.77)
    assert list(result) == [False, True, False]


def test_temperate_mask_central_boundary_inclusive():
    flux = _flux([0.25, 1.77])
    result = temperate_mask(flux, None, None, lower=0.25, upper=1.77)
    assert all(result)


def test_temperate_mask_excludes_masked_flux():
    flux = _flux([1.0, 1.0], mask=[True, False])
    result = temperate_mask(flux, None, None, lower=0.25, upper=1.77)
    assert list(result) == [False, True]


def test_temperate_mask_interval_includes_edge_overlap():
    # Central flux 2.0 is outside [0.25, 1.77]; with err2=0.3, lower edge = 1.7 < 1.77
    flux = _flux([2.0])
    err2 = _flux([0.3])   # lower 1σ: flux - err2 = 1.7 <= upper=1.77
    assert not temperate_mask(flux, None, None, lower=0.25, upper=1.77)[0]  # central
    assert     temperate_mask(flux, None, err2, lower=0.25, upper=1.77, use_interval=True)[0]


def test_temperate_mask_interval_no_overlap():
    # Central=2.5, err1=err2=0.1 → [2.4, 2.6] does not overlap [0.25, 1.77]
    flux = _flux([2.5])
    err1 = _flux([0.1])
    err2 = _flux([0.1])
    assert not temperate_mask(flux, err1, err2, lower=0.25, upper=1.77, use_interval=True)[0]


def test_temperate_mask_masked_err_falls_back_to_central():
    # masked err2 → treat as 0 → fall back to central-value test
    flux = _flux([2.0])
    err2 = MaskedColumn([0.5], mask=[True], name='pl_insol_err2')
    # Central 2.0 > 1.77 → False, even though unmasked err2=0.5 would overlap
    assert not temperate_mask(flux, None, err2, lower=0.25, upper=1.77, use_interval=True)[0]


# --- mass_radius_chen_kipping ---

def test_mass_radius_zero_guard():
    assert mass_radius_chen_kipping(0.0) == 0.0


def test_mass_radius_negative_guard():
    assert mass_radius_chen_kipping(-1.0) == 0.0


def test_mass_radius_terran():
    # 1 M_Earth → R = 1.008 × 1^0.279 = 1.008
    assert mass_radius_chen_kipping(1.0) == pytest.approx(1.008)


def test_mass_radius_neptunian():
    # 10 M_Earth → R = 0.808 × 10^0.589
    expected = 0.808 * 10 ** 0.589
    assert mass_radius_chen_kipping(10.0) == pytest.approx(expected, rel=1e-4)


def test_mass_radius_boundary_continuous():
    # Both branches should agree at 2.04 M_Earth to < 0.1%
    terran    = 1.008 * 2.04 ** 0.279
    neptunian = 0.808 * 2.04 ** 0.589
    assert abs(terran - neptunian) / terran < 0.001


# --- rocky_mask ---

def _r(vals, mask=None):
    if mask is None:
        mask = [False] * len(vals)
    return MaskedColumn(vals, mask=mask, name='r_earth')


def _rb(vals, mask=None):
    """Build a MaskedColumn for r_earth_min / r_earth_max."""
    if mask is None:
        mask = [False] * len(vals)
    return MaskedColumn(vals, mask=mask, name='r_bound')


def test_rocky_mask_in_range():
    r = _r([1.0])
    assert rocky_mask(r, None, None, lower=0.5, upper=1.5)[0]


def test_rocky_mask_out_of_range():
    r = _r([2.0])
    assert not rocky_mask(r, None, None, lower=0.5, upper=1.5)[0]


def test_rocky_mask_excludes_masked_r():
    r = _r([1.0], mask=[True])
    assert not rocky_mask(r, None, None, lower=0.5, upper=1.5)[0]


def test_rocky_mask_interval_catches_uncertain():
    # r_earth masked; rmin=1.1, rmax=1.4 → overlaps [0.5, 1.5]
    r    = _r([0.0], mask=[True])
    rmin = _rb([1.1])
    rmax = _rb([1.4])
    assert rocky_mask(r, rmin, rmax, lower=0.5, upper=1.5, use_interval=True)[0]


def test_rocky_mask_interval_no_overlap():
    # r_earth masked; rmin=2.0, rmax=3.0 → entirely outside [0.5, 1.5]
    r    = _r([0.0], mask=[True])
    rmin = _rb([2.0])
    rmax = _rb([3.0])
    assert not rocky_mask(r, rmin, rmax, lower=0.5, upper=1.5, use_interval=True)[0]


def test_rocky_mask_no_bounds_falls_back():
    # r_earth masked, no bounds → use_interval=True still False
    r = _r([0.0], mask=[True])
    assert not rocky_mask(r, None, None, lower=0.5, upper=1.5, use_interval=True)[0]


def test_rocky_mask_interval_false_ignores_bounds():
    # use_interval=False: even with rmin/rmax in range, masked r → False
    r    = _r([0.0], mask=[True])
    rmin = _rb([1.0])
    rmax = _rb([1.3])
    assert not rocky_mask(r, rmin, rmax, lower=0.5, upper=1.5, use_interval=False)[0]


def test_rocky_mask_known_rocky_still_included_in_interval_mode():
    # use_interval=True should also include confirmed rocky planets
    r = _r([1.0])
    assert rocky_mask(r, None, None, lower=0.5, upper=1.5, use_interval=True)[0]


def test_temperate_mask_none_err_in_interval_mode():
    # Both err1=None and err2=None → same as central test
    flux = _flux([1.0, 2.0])
    result = temperate_mask(flux, None, None, lower=0.25, upper=1.77, use_interval=True)
    assert list(result) == [True, False]


# --- _mann_teff_radius ---

def test_mann_teff_solar_type_star():
    # Teff = 3500 K (x=1) → polynomial collapses to sum of coefficients
    # 10.5440 - 33.7546 + 35.1909 - 11.5928 ≈ 0.387 R_sun (M2 dwarf)
    r = _mann_teff_radius(3500.0)
    assert 0.35 < r < 0.45

def test_mann_teff_late_m_dwarf():
    # Teff = 3000 K — late M dwarf, R ≈ 0.15–0.20 R_sun
    r = _mann_teff_radius(3000.0)
    assert 0.10 < r < 0.25

def test_mann_teff_zero_guard():
    assert _mann_teff_radius(0.0) == 0.0  # invalid input → 0

def test_mann_teff_metallicity_increases_radius():
    r_solar = _mann_teff_radius(3500.0, met=0.0)
    r_supersolar = _mann_teff_radius(3500.0, met=0.3)
    assert r_supersolar > r_solar

def test_mann_teff_none_met_equals_solar():
    assert _mann_teff_radius(3500.0, met=None) == pytest.approx(_mann_teff_radius(3500.0, met=0.0))


# --- _mann_mks_radius ---

def test_mann_mks_early_m_dwarf():
    # M_Ks ≈ 5.0 (early M dwarf) → R ≈ 0.45–0.65 R_sun
    r = _mann_mks_radius(5.0)
    assert 0.40 < r < 0.70

def test_mann_mks_mid_m_dwarf():
    # M_Ks ≈ 8.5 (mid M dwarf) → R ≈ 0.15–0.30 R_sun
    r = _mann_mks_radius(8.5)
    assert 0.10 < r < 0.35

def test_mann_mks_zero_guard():
    # Extremely bright M_Ks = 0 → polynomial may give implausible value, clamped to 0.05
    r = _mann_mks_radius(0.0)
    assert r >= 0.05

def test_mann_mks_none_met_equals_solar():
    assert _mann_mks_radius(6.0, met=None) == pytest.approx(_mann_mks_radius(6.0, met=0.0))


# --- _torres_radius ---

def test_torres_solar_star():
    # Sun: Teff=5778, logg=4.438, [Fe/H]=0 → R ≈ 1.0 R_sun
    r = _torres_radius(5778.0, 4.438, 0.0)
    assert 0.95 < r < 1.05

def test_torres_f_star():
    # F5V dwarf: Teff=6500, logg=4.2, [Fe/H]=0 → R ≈ 1.3–1.8 R_sun
    r = _torres_radius(6500.0, 4.2, 0.0)
    assert 1.2 < r < 1.9

def test_torres_k_dwarf():
    # K5 dwarf: Teff=4700, logg=4.7, [Fe/H]=0 → R ≈ 0.6–0.9 R_sun
    r = _torres_radius(4700.0, 4.7, 0.0)
    assert 0.55 < r < 0.95

def test_torres_none_met_treated_as_solar():
    assert _torres_radius(5778.0, 4.438, None) == pytest.approx(
        _torres_radius(5778.0, 4.438, 0.0), rel=1e-6
    )

def test_torres_metallicity_effect():
    r_solar    = _torres_radius(5778.0, 4.438, 0.0)
    r_metpoor  = _torres_radius(5778.0, 4.438, -0.5)
    r_metrich  = _torres_radius(5778.0, 4.438,  0.5)
    # Metal-rich stars are slightly larger; metal-poor slightly smaller
    assert r_metrich > r_solar > r_metpoor


# --- Source error propagation (regression for suffix-mismatch bug) ---
#
# Parameters without inference fallbacks (logg, met, vmag, dist) can only
# receive errors from _merge_values.  The bug was that _merge_values looked up
# errors using the *table-level* suffix (e.g. "_max") rather than the source
# dict's internal convention ("err1"/"err2"), so errors were silently dropped.

import numpy as np
import pytest
from astropy.table import Table
from tests.enrich_keys import DEFAULT_ENRICH_KEYS
from crossmatching.enrichment import ParamFiller
from crossmatching.enrichment.param_sources.base import ParamSource


class _FixedParamSource(ParamSource):
    """Minimal toy source returning pre-specified params for a single key."""

    key_col = "main_id"
    source_name = "test_src"

    def __init__(self, data: dict):
        self._lookup = data

    def download(self, key_list):
        raise NotImplementedError

    def _build_lookup(self, table):
        return {}


@pytest.fixture(scope="module")
def _enriched_with_source_errors():
    """Enrich a one-row table whose source supplies errors for logg/met/vmag/dist."""
    src = _FixedParamSource({
        "HD 1234": {
            "teff": 5500.0, "tefferr1": 80.0, "tefferr2": 80.0,
            "rad":  1.05,   "raderr1":  0.03, "raderr2":  0.03,
            "logg": 4.40,   "loggerr1": 0.05, "loggerr2": 0.05,
            "met":  0.10,   "meterr1":  0.02, "meterr2":  0.02,
            "vmag": 8.50,   "vmagerr1": 0.01, "vmagerr2": 0.01,
            "dist": 45.0,   "disterr1": 1.0,  "disterr2": 1.0,
        }
    })

    table = Table({
        "main_id": ["HD 1234"],
        "st_rad":  [np.nan],
        "st_mass": [np.nan],
        "st_teff": [np.nan],
        "st_logg": [np.nan],
        "st_met":  [np.nan],
        "sy_vmag": [np.nan],
        "sy_dist": [np.nan],
        "r":       [np.nan],
        "a":       [np.nan],
        "p":       [np.nan],
        "msini":   [np.nan],
    })

    merger = ParamFiller([src])
    return merger.enrich(table, **DEFAULT_ENRICH_KEYS, disable_calculations=True)


@pytest.mark.parametrize("col,expected_err", [
    ("st_logg", 0.05),
    ("st_met",  0.02),
    ("sy_vmag", 0.01),
    ("sy_dist", 1.0),
    ("st_teff", 80.0),
])
def test_source_errors_propagated_to_output(col, expected_err, _enriched_with_source_errors):
    """Source-provided errors must appear in both upper and lower error columns."""
    out = _enriched_with_source_errors
    upper_col = f"{col}err1"
    lower_col  = f"{col}err2"
    assert upper_col in out.colnames, f"Missing column {upper_col}"
    assert lower_col  in out.colnames, f"Missing column {lower_col}"
    assert not np.ma.getmaskarray(out[upper_col])[0], f"{upper_col} is masked (no error propagated)"
    assert not np.ma.getmaskarray(out[lower_col])[0],  f"{lower_col} is masked (no error propagated)"
    assert float(out[upper_col][0]) == pytest.approx(expected_err, rel=1e-6)
    assert float(out[lower_col][0]) == pytest.approx(expected_err, rel=1e-6)


def test_source_with_custom_error_suffix(monkeypatch):
    """A source declaring a non-default error_upper/lower_suffix is respected."""
    src = _FixedParamSource({
        "HD 9999": {
            "logg": 4.30,
            "logg_upper": 0.08,  # custom suffix
            "logg_lower": 0.06,
        }
    })
    monkeypatch.setattr(src, "error_upper_suffix", "_upper")
    monkeypatch.setattr(src, "error_lower_suffix", "_lower")

    table = Table({
        "main_id": ["HD 9999"],
        "st_logg": [np.nan],
        "st_rad":  [np.nan], "st_mass": [np.nan], "st_teff": [np.nan],
        "st_met":  [np.nan], "sy_vmag": [np.nan], "sy_dist": [np.nan],
        "r": [np.nan], "a": [np.nan], "p": [np.nan], "msini": [np.nan],
    })

    out = ParamFiller([src]).enrich(table, **DEFAULT_ENRICH_KEYS, disable_calculations=True)
    assert float(out["st_loggerr1"][0]) == pytest.approx(0.08, rel=1e-6)
    assert float(out["st_loggerr2"][0]) == pytest.approx(0.06, rel=1e-6)

