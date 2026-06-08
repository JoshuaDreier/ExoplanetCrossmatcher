import numpy as np
import pytest
from astropy.table import MaskedColumn

from crossmatching.enrichment import (
    classify_spectral_type,
    ms_radius_from_teff,
    spectype_display,
    temperate_mask,
    teff_to_spectype,
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
    return MaskedColumn(vals, mask=mask, name='flux_rel')


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
    err2 = MaskedColumn([0.5], mask=[True], name='flux_rel_err2')
    # Central 2.0 > 1.77 → False, even though unmasked err2=0.5 would overlap
    assert not temperate_mask(flux, None, err2, lower=0.25, upper=1.77, use_interval=True)[0]


def test_temperate_mask_none_err_in_interval_mode():
    # Both err1=None and err2=None → same as central test
    flux = _flux([1.0, 2.0])
    result = temperate_mask(flux, None, None, lower=0.25, upper=1.77, use_interval=True)
    assert list(result) == [True, False]
