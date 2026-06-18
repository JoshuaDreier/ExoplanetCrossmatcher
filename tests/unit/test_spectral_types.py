"""Unit tests for spectral_types.py

Covers:
  - standardize_spectral_type  – public normalisation API
  - _parse_spt                 – float-index mapper (uses standardize internally)
  - spectype_to_teff           – interpolated Teff lookup
  - get_spectral_class_range   – Teff range for uncertainty estimation
"""
from __future__ import annotations

import pytest

from crossmatching.enrichment.spectral_types import (
    standardize_spectral_type,
    _parse_spt,
    spectype_to_teff,
    get_spectral_class_range,
    _TEFF_SPECTYPE,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _idx(letter: str, subtype: float) -> float:
    """Expected float index for a letter + subtype pair."""
    order = "OBAFGKMLTY"
    return order.index(letter) * 10 + subtype


# ===========================================================================
# standardize_spectral_type  – simple / clean inputs
# ===========================================================================

def test_standardize_plain_letter_and_subtype():
    assert standardize_spectral_type("G2V") == "G2"

def test_standardize_fractional_subtype():
    assert standardize_spectral_type("M3.5III") == "M3.5"

def test_standardize_letter_only():
    assert standardize_spectral_type("K") == "K"

def test_standardize_luminosity_class_stripped():
    assert standardize_spectral_type("B8IV") == "B8"

def test_standardize_peculiarity_flag_stripped():
    assert standardize_spectral_type("A0VpSiEu") == "A0"

def test_standardize_uncertainty_colon_stripped():
    assert standardize_spectral_type("G3III:") == "G3"

def test_standardize_simple_O_type():
    assert standardize_spectral_type("O6") == "O6"

def test_standardize_simple_B_type():
    assert standardize_spectral_type("B3V") == "B3"

def test_standardize_simple_A_type():
    assert standardize_spectral_type("A1IV") == "A1"

def test_standardize_simple_F_type():
    assert standardize_spectral_type("F5V") == "F5"


# ===========================================================================
# standardize_spectral_type  – metallic-line (Am) notation
# ===========================================================================

def test_standardize_am_khm_basic():
    """kXhYmZ: hydrogen class Y is returned."""
    assert standardize_spectral_type("kA4hA5mA5Va") == "A5"

def test_standardize_am_khm_cross_class():
    """Hydrogen class may differ significantly from k-line class."""
    assert standardize_spectral_type("kA5hF0mF2III") == "F0"

def test_standardize_am_khm_fractional():
    assert standardize_spectral_type("kA2hA5mA4IV-V") == "A5"

def test_standardize_am_khm_lum_suffix_before_h():
    """Luminosity suffix on k-type (e.g. kF3V h…) must be skipped."""
    assert standardize_spectral_type("kF3VhF5mF5(II-III)") == "F5"

def test_standardize_am_plain_base_returned():
    """A0mA1Va: base spectral type is returned (no k prefix)."""
    assert standardize_spectral_type("A0mA1Va") == "A0"

def test_standardize_am_F0mF2_plain():
    assert standardize_spectral_type("F0mF2V") == "F0"

def test_standardize_am_kA2hF2mF2_paren():
    assert standardize_spectral_type("kA2hF2mF2(IV)") == "F2"

def test_standardize_am_kA3hA5mA5():
    assert standardize_spectral_type("kA3hA5mA5IV-V") == "A5"

def test_standardize_am_kA5hF0mF2():
    assert standardize_spectral_type("kA5hF0mF2III") == "F0"


# ===========================================================================
# standardize_spectral_type  – subdwarf / dwarf prefixes
# ===========================================================================

def test_standardize_sd_prefix_stripped():
    assert standardize_spectral_type("sdM3.5") == "M3.5"

def test_standardize_esd_prefix_stripped():
    assert standardize_spectral_type("esdM0") == "M0"

def test_standardize_d_prefix_stripped():
    assert standardize_spectral_type("dM4.5e") == "M4.5"

def test_standardize_sd_colon_separator():
    assert standardize_spectral_type("sd:K1Fe-1") == "K1"

def test_standardize_sdK_type():
    assert standardize_spectral_type("sdK7") == "K7"


# ===========================================================================
# standardize_spectral_type  – binary / composite types
# ===========================================================================

def test_standardize_binary_plus_keeps_primary():
    assert standardize_spectral_type("G9III+A7.5") == "G9"

def test_standardize_binary_slash_keeps_primary():
    assert standardize_spectral_type("M2.5V/M3V") == "M2.5"

def test_standardize_binary_DQ_secondary_ignored():
    """White-dwarf secondary (DQZ) should not pollute the result."""
    assert standardize_spectral_type("F5IV-V+DQZ") == "F5"

def test_standardize_binary_two_regular():
    assert standardize_spectral_type("F7V+G4V") == "F7"

def test_standardize_binary_primary_no_subtype_secondary():
    assert standardize_spectral_type("K1V+G") == "K1"

def test_standardize_slash_ambiguous_subtype():
    """G3/6 – ambiguous subtype range; primary component G3 is used."""
    assert standardize_spectral_type("G3/6") == "G3"


# ===========================================================================
# standardize_spectral_type  – parenthesised qualifiers
# ===========================================================================

def test_standardize_paren_n_removed():
    assert standardize_spectral_type("A5IV(n)") == "A5"

def test_standardize_paren_e_removed():
    assert standardize_spectral_type("M3V(e)") == "M3"

def test_standardize_paren_luminosity_removed():
    assert standardize_spectral_type("kF3VhF5mF5(II-III)") == "F5"


# ===========================================================================
# standardize_spectral_type  – white-dwarf types → empty string
# ===========================================================================

def test_standardize_white_dwarf_DA_returns_empty():
    assert standardize_spectral_type("DA2") == ""

def test_standardize_white_dwarf_DQ_returns_empty():
    assert standardize_spectral_type("DQ") == ""

def test_standardize_white_dwarf_DA_with_secondary_partner():
    """Primary is a WD – the whole string maps to empty."""
    assert standardize_spectral_type("DA3+F6V") == ""


# ===========================================================================
# standardize_spectral_type  – unparseable / degenerate inputs
# ===========================================================================

def test_standardize_null_string_returns_empty():
    assert standardize_spectral_type("null") == ""

def test_standardize_empty_string_returns_empty():
    assert standardize_spectral_type("") == ""

def test_standardize_nan_string_returns_empty():
    assert standardize_spectral_type("nan") == ""

def test_standardize_OBepec_reduces_to_O():
    """'OBepec' – only the first letter class is extractable."""
    assert standardize_spectral_type("OBepec") == "O"


# ===========================================================================
# standardize_spectral_type  – full catalogue parametrised cases
# ===========================================================================

@pytest.mark.parametrize("raw,expected", [
    # B
    ("B6Vpe",           "B6"),
    ("B8IVn",           "B8"),
    ("B8IV-VHgMn",      "B8"),
    ("B6V",             "B6"),
    ("B8V",             "B8"),
    ("B3V",             "B3"),
    ("B9IVp",           "B9"),
    # A
    ("A0mA1Va",         "A0"),
    ("A1IV-Vp",         "A1"),
    ("A1.5IV+",         "A1.5"),
    ("A1IV+A0IV",       "A1"),
    ("A5IV(n)",         "A5"),
    ("A2.5Va",          "A2.5"),
    ("kA4hA5mA5Va",     "A5"),
    ("kA2hA5mA4IV-V",   "A5"),
    ("A3IV",            "A3"),
    ("A0VpSiEu",        "A0"),
    ("A5IVs",           "A5"),
    ("A4IV/V",          "A4"),
    ("A5IVnn",          "A5"),
    ("A8V+F3V",         "A8"),
    ("A7V:",            "A7"),
    ("A1.5Vas",         "A1.5"),
    ("A2IV+A4V",        "A2"),
    ("A1V+A2Vm",        "A1"),
    ("A1IV",            "A1"),
    ("kA2hF2mF2(IV)",   "F2"),
    ("kA3hA5mA5IV-V",   "A5"),
    ("kA5hF0mF2III",    "F0"),
    ("F0mF2V",          "F0"),
    # F
    ("F1IV-V(n)",       "F1"),
    ("F5V+F6V",         "F5"),
    ("F9VFe-1.7CH-0.7", "F9"),
    ("G0Vn",            "G0"),
    ("F7V+G4V",         "F7"),
    ("F:",              "F"),
    # G
    ("G2IV+G2IV",       "G2"),
    ("G3V+K0V",         "G3"),
    ("G8VFe+0.5",       "G8"),
    ("G2VFe-3",         "G2"),
    ("G5Vb",            "G5"),
    ("G2V+G2V",         "G2"),
    ("G8V+G9V",         "G8"),
    # K
    ("K1.5IIIFe-0.5",   "K1.5"),
    ("K5+III",          "K5"),
    ("K5III",           "K5"),
    ("K2-IIIbCa-1",     "K2"),
    ("K1-III+G7IIIb",   "K1"),
    ("K4-III",          "K4"),
    ("K0-IIIb",         "K0"),
    ("K0.5IIIb",        "K0.5"),
    ("K1V+K4V",         "K1"),
    ("K7.0Ve",          "K7.0"),
    ("K2IIIbCN1",       "K2"),
    ("K1IIIb",          "K1"),
    # M
    ("M3.5III",         "M3.5"),
    ("M0.5III",         "M0.5"),
    ("M3.85",           "M3.85"),
    ("M3.14",           "M3.14"),
    ("M4.20",           "M4.20"),
    ("M4.01",           "M4.01"),
    ("M3.74",           "M3.74"),
    ("M4.32",           "M4.32"),
    ("sdM3.5",          "M3.5"),
    ("M2.5V/M3V",       "M2.5"),
    ("M4.5+M4.5",       "M4.5"),
    ("M2+Vk",           "M2"),
    ("M2III",           "M2"),
])
def test_standardize_catalogue_cases(raw, expected):
    assert standardize_spectral_type(raw) == expected


# ===========================================================================
# _parse_spt  – simple clean inputs
# ===========================================================================

def test_parse_spt_anchor_G2():
    assert _parse_spt("G2V") == pytest.approx(_idx("G", 2.0))

def test_parse_spt_anchor_M0():
    assert _parse_spt("M0") == pytest.approx(_idx("M", 0.0))

def test_parse_spt_anchor_O5_5():
    assert _parse_spt("O5.5") == pytest.approx(_idx("O", 5.5))

def test_parse_spt_letter_only_defaults_to_5():
    assert _parse_spt("K") == pytest.approx(_idx("K", 5.0))


# ===========================================================================
# _parse_spt  – complex inputs now accepted
# ===========================================================================

def test_parse_spt_am_kA4hA5mA5Va():
    """kA4hA5mA5Va → standardized A5 → A rank * 10 + 5."""
    assert _parse_spt("kA4hA5mA5Va") == pytest.approx(_idx("A", 5.0))

def test_parse_spt_am_kA5hF0mF2III():
    assert _parse_spt("kA5hF0mF2III") == pytest.approx(_idx("F", 0.0))

def test_parse_spt_sd_prefix():
    assert _parse_spt("sdM3.5") == pytest.approx(_idx("M", 3.5))

def test_parse_spt_binary_plus():
    assert _parse_spt("G9III+A7.5") == pytest.approx(_idx("G", 9.0))

def test_parse_spt_binary_slash():
    assert _parse_spt("M2.5V/M3V") == pytest.approx(_idx("M", 2.5))

def test_parse_spt_paren_qualifier():
    assert _parse_spt("A1IV(n)") == pytest.approx(_idx("A", 1.0))

def test_parse_spt_luminosity_class_suffix_ignored():
    assert _parse_spt("K5III") == pytest.approx(_idx("K", 5.0))

def test_parse_spt_hyphen_lum_class():
    assert _parse_spt("K4-III") == pytest.approx(_idx("K", 4.0))

def test_parse_spt_peculiarity_flags_ignored():
    assert _parse_spt("A0VpSiEu") == pytest.approx(_idx("A", 0.0))

def test_parse_spt_fractional_subtype_no_anchor():
    """Fractional subtype not in anchor table still parses correctly."""
    assert _parse_spt("K2.5V") == pytest.approx(_idx("K", 2.5))

def test_parse_spt_Fe_suffix_ignored():
    assert _parse_spt("K1.5IIIFe-0.5") == pytest.approx(_idx("K", 1.5))


# ===========================================================================
# _parse_spt  – white-dwarf / unparseable raises ValueError
# ===========================================================================

def test_parse_spt_white_dwarf_raises():
    with pytest.raises(ValueError):
        _parse_spt("DA2")

def test_parse_spt_empty_string_raises():
    with pytest.raises(ValueError):
        _parse_spt("")

def test_parse_spt_null_raises():
    with pytest.raises(ValueError):
        _parse_spt("null")

def test_parse_spt_DQ_raises():
    with pytest.raises(ValueError):
        _parse_spt("DQ")


# ===========================================================================
# spectype_to_teff
# ===========================================================================

def test_spectype_to_teff_anchor_G2():
    assert spectype_to_teff("G2V") == pytest.approx(5780.0)

def test_spectype_to_teff_anchor_M0():
    assert spectype_to_teff("M0") == pytest.approx(3850.0)

def test_spectype_to_teff_interpolated_K2_5():
    """K2.5 is between K2 (5040 K) and K3 (4830 K)."""
    teff = spectype_to_teff("K2.5V")
    assert 4830.0 < teff < 5040.0

def test_spectype_to_teff_complex_am_type():
    """kA4hA5mA5Va standardises to A5 → 8180 K."""
    assert spectype_to_teff("kA4hA5mA5Va") == pytest.approx(8180.0)

def test_spectype_to_teff_binary_primary_used():
    """G9III+A7.5 → G9 → between G8 (5440 K) and K0 (5280 K)."""
    teff = spectype_to_teff("G9III+A7.5")
    assert 5280.0 < teff < 5440.0

def test_spectype_to_teff_sd_prefix_ignored():
    assert spectype_to_teff("sdM3.5") == pytest.approx(spectype_to_teff("M3.5"))

def test_spectype_to_teff_white_dwarf_raises():
    with pytest.raises(ValueError):
        spectype_to_teff("DA2")


# ===========================================================================
# get_spectral_class_range
# ===========================================================================

def test_range_anchor_G2():
    lo, hi = get_spectral_class_range("G2V")
    assert lo == pytest.approx(5660.0)   # G5
    assert hi == pytest.approx(5860.0)   # G1

def test_range_anchor_M0():
    lo, hi = get_spectral_class_range("M0")
    assert lo == pytest.approx(3660.0)   # M1 is the next-cooler anchor
    assert hi == pytest.approx(3930.0)   # K9 is the next-hotter anchor above M0

def test_range_anchor_O3_extrapolates_upper():
    """O3 is the hottest anchor; t_max is extrapolated above it."""
    lo, hi = get_spectral_class_range("O3")
    assert hi > 44200.0

def test_range_anchor_M9_extrapolates_lower():
    """M9 is the coolest anchor; t_min is extrapolated below it."""
    lo, hi = get_spectral_class_range("M9")
    assert lo < 2400.0

def test_range_non_anchor_subtype_falls_back_to_class():
    """K2.5 has no exact anchor; range falls back to the full K-class span."""
    lo, hi = get_spectral_class_range("K2.5V")
    k_lo, k_hi = get_spectral_class_range("K")
    assert lo == pytest.approx(k_lo)
    assert hi == pytest.approx(k_hi)

def test_range_broad_M_class():
    lo, hi = get_spectral_class_range("M")
    assert lo == pytest.approx(2400.0)   # M9
    assert hi == pytest.approx(3850.0)   # M0

def test_range_broad_K_class():
    lo, hi = get_spectral_class_range("K")
    assert lo == pytest.approx(3930.0)   # K9
    assert hi == pytest.approx(5280.0)   # K0

def test_range_am_type_equals_standardized():
    """kA4hA5mA5Va → A5 → same range as plain A5."""
    assert get_spectral_class_range("kA4hA5mA5Va") == get_spectral_class_range("A5")

def test_range_sd_type_equals_plain():
    assert get_spectral_class_range("sdM3.5") == get_spectral_class_range("M3.5")

def test_range_binary_uses_primary():
    assert get_spectral_class_range("G9III+A7.5") == get_spectral_class_range("G9")

def test_range_luminosity_suffix_ignored():
    assert get_spectral_class_range("K5III") == get_spectral_class_range("K5")

def test_range_white_dwarf_returns_zero():
    assert get_spectral_class_range("DA2") == (0.0, 0.0)

def test_range_empty_string_returns_zero():
    assert get_spectral_class_range("") == (0.0, 0.0)

def test_range_null_string_returns_zero():
    assert get_spectral_class_range("null") == (0.0, 0.0)

def test_range_lo_le_hi_for_all_anchors():
    """min Teff ≤ max Teff for every anchor in the table."""
    for spt, _ in _TEFF_SPECTYPE:
        lo, hi = get_spectral_class_range(spt)
        assert lo <= hi, f"Range inverted for {spt}: ({lo}, {hi})"
