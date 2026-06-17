"""Integration tests for StellarParamMerger with toy source tables and uncertainty propagation.

Four planets exercise the full tier precedence and msini radius estimation:
  Planet A — present in all three sources; HPIC wins for stellar params, NEA fills insol/mass
  Planet B — NEA only
  Planet C — SIMBAD only; rad derived from Mann 2015 Teff polynomial (teff < 4000K)
  Planet D — msini only, no direct radius; tests uncertain-rocky estimation
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.table import MaskedColumn, Table

from crossmatching.enrichment import (
    R_JUP_TO_EARTH,
    ParamFiller,
    _mann_teff_radius,
    mass_radius_chen_kipping,
    rocky_mask,
    temperate_mask,
)
from crossmatching.enrichment.param_sources.hpic import HpicParamSource
from crossmatching.enrichment.param_sources.nea import NeaParamSource
from crossmatching.enrichment.param_sources.simbad import SimbadParamSource

_MSINI_D = 1.27 / u.M_jup.to(u.M_earth)  # 1.27 M_Earth expressed in M_Jup


# ── toy data ──────────────────────────────────────────────────────────────────

def _hpic_table():
    return Table({
        "exo-mercat_name": ["Planet A"],
        "st_teff":         [5600.0],
        "st_rad":          [0.95],
        "st_spectype":     ["G5V"],
        "sy_vmag":         [7.1],
    })


def _nea_table():
    return Table({
        "pl_name":     ["Planet A", "Planet B"],
        "st_teff":     [5200.0,     4800.0],
        "st_tefferr1": [  80.0,       60.0],
        "st_tefferr2": [ -70.0,      -50.0],
        "st_rad":      [0.80,       0.65],
        "st_raderr1":  [0.04,       0.03],
        "st_raderr2":  [-0.03,     -0.02],
        "st_mass":     [1.1,        0.75],
        "st_masserr1": [0.05,       0.04],
        "st_masserr2": [-0.04,     -0.03],
        "st_spectype": ["K0V",      "K3V"],
        "pl_insol":    [1.2,        0.55],
        "pl_insolerr1":[0.1,        0.05],
        "pl_insolerr2":[-0.08,     -0.04],
        "sy_vmag":     [7.9,        9.2],
        "sy_vmagerr1": [0.02,       0.03],
        "sy_vmagerr2": [-0.02,     -0.03],
        "sy_dist":     [12.0,       20.0],
        "sy_disterr1": [0.5,        1.0],
        "sy_disterr2": [-0.4,      -0.9],
        "st_logg":     [4.4,        4.6],
        "st_loggerr1": [0.1,        0.08],
        "st_loggerr2": [-0.09,     -0.07],
        "st_met":      [0.1,       -0.1],
        "st_meterr1":  [0.05,       0.04],
        "st_meterr2":  [-0.05,     -0.04],
        "pl_eqt":      [np.nan,    np.nan],
        "pl_eqterr1":  [np.nan,    np.nan],
        "pl_eqterr2":  [np.nan,    np.nan],
    })


def _simbad_table():
    return Table({
        "main_id":   ["simbad_host_C"],
        "teff":      [3400.0],
        "sp_type":   ["M2V"],
        "plx_value": [100.0],  # 10 pc
        "vmag":      [np.nan],
    })


def _catalog_table():
    """Minimal EMC-shaped table with 4 test planets.

    Planet D has no direct radius but has msini — exercises uncertain-rocky estimation.
    """
    return Table({
        "exo-mercat_name": ["Planet A", "Planet B", "Planet C", "Planet D"],
        "nasa_name":       ["Planet A", "Planet B", "unknown",  ""],
        "main_id":         ["main_A",   "main_B",   "simbad_host_C", ""],
        "r":               [0.1,        0.2,        0.08,       0.0],    # Jupiter radii; D absent
        "a":               [1.0,        0.0,        0.0,        0.0485], # AU
        "p":               [365.0,      30.0,       0.0,        11.186], # days
        "msini":           [np.nan,     np.nan,     np.nan,     _MSINI_D],  # M_Jup; D only
    })


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def enriched():
    hpic_src = HpicParamSource(_hpic_table())
    hpic_src.load()

    nea_src = NeaParamSource()
    nea_src._lookup = nea_src._build_lookup(_nea_table())

    simbad_src = SimbadParamSource()
    simbad_src._lookup = simbad_src._build_lookup(_simbad_table())

    merger = ParamFiller([hpic_src, nea_src, simbad_src])
    return merger.enrich(_catalog_table())


def _row(table, name):
    idx = list(table["exo-mercat_name"]).index(name)
    return table[idx]


# ── expected columns present ──────────────────────────────────────────────────

def test_all_enriched_columns_present(enriched):
    for col in (
        "st_teff", "st_rad", "st_mass", "sy_vmag", "sy_kmag", "sy_dist",
        "st_logg", "st_met", "st_lum", "pl_eqt",
        "st_spectype", "st_spectype_src",
        "r_lower_bound", "r_upper_bound", "pl_insol", "spectral_category",
        "st_teff_src", "st_rad_src", "st_mass_src", "sy_vmag_src", "sy_kmag_src",
        "sy_dist_src", "st_logg_src", "st_met_src", "st_lum_src", "pl_eqt_src",
        "a_src", "pl_insol_src",
        "st_teff_err1", "st_teff_err2", "st_rad_err1", "st_rad_err2",
        "st_mass_err1", "st_mass_err2", "sy_vmag_err1", "sy_vmag_err2",
        "sy_kmag_err1", "sy_kmag_err2",
        "sy_dist_err1", "sy_dist_err2", "st_logg_err1", "st_logg_err2",
        "st_met_err1",  "st_met_err2",  "st_lum_err1",  "st_lum_err2",
        "pl_eqt_err1",  "pl_eqt_err2",  "pl_insol_err1", "pl_insol_err2",
    ):
        assert col in enriched.colnames, f"missing column: {col}"


def test_enrich_handles_missing_planet_columns():
    """Tables without the EMC planet columns (r/a/p/msini) must not raise;

    dependent outputs stay masked while direct source values still bind.
    """
    nea = NeaParamSource()
    nea._lookup = nea._build_lookup(_nea_table())
    catalog = Table({
        "exo-mercat_name": ["Planet A"],
        "nasa_name":       ["Planet A"],
        "main_id":         [""],
    })
    result = ParamFiller([nea]).enrich(catalog)
    assert np.ma.is_masked(result["r_lower_bound"][0])
    assert np.ma.is_masked(result["r_upper_bound"][0])
    assert str(result["a_src"][0]) == ""
    assert str(result["pl_insol_src"][0]) == "insol:nea"  # direct insol still binds


# ── Planet A: HPIC wins for stellar params; NEA fills mass / insol ─────────

def test_planet_a_hpic_teff_wins(enriched):
    row = _row(enriched, "Planet A")
    assert float(row["st_teff"]) == pytest.approx(5600.0)   # HPIC, not NEA 5200


def test_planet_a_hpic_rad_wins(enriched):
    row = _row(enriched, "Planet A")
    assert float(row["st_rad"]) == pytest.approx(0.95)


def test_planet_a_nea_fills_mass(enriched):
    row = _row(enriched, "Planet A")
    assert float(row["st_mass"]) == pytest.approx(1.1)      # NEA (HPIC had no mass)


def test_planet_a_pl_insol_unmasked(enriched):
    row = _row(enriched, "Planet A")
    assert not np.ma.is_masked(row["pl_insol"])
    assert float(row["pl_insol"]) > 0


# ── Source provenance ────────────────────────────────────────────────────────

def test_planet_a_teff_src_is_hpic(enriched):
    row = _row(enriched, "Planet A")
    assert str(row["st_teff_src"]) == "hpic"


def test_planet_a_mass_src_is_nea(enriched):
    row = _row(enriched, "Planet A")
    assert str(row["st_mass_src"]) == "nea"


def test_planet_a_pl_insol_src_is_insol_nea(enriched):
    # Planet A has pl_insol from NEA → pl_insol taken directly
    row = _row(enriched, "Planet A")
    assert str(row["pl_insol_src"]) == "insol:nea"


def test_planet_b_a_src_is_kepler(enriched):
    # Planet B has a=0 in catalog, so Kepler fallback is used; mass from NEA
    row = _row(enriched, "Planet B")
    assert str(row["a_src"]).startswith("kepler(mass:nea")


def test_planet_c_rad_src_is_ms_from_simbad(enriched):
    # Planet C: SIMBAD provides teff=3400K; mann_teff fallback applies (< 4000K, no K-band)
    row = _row(enriched, "Planet C")
    assert str(row["st_rad_src"]) == "mann_teff(teff:simbad)"


def test_planet_c_pl_insol_src_empty(enriched):
    # Planet C has no a, no insol → flux masked, src empty
    row = _row(enriched, "Planet C")
    assert str(row["pl_insol_src"]) == ""


def test_planet_b_pl_insol_src_is_direct_insol(enriched):
    # Planet B has pl_insol from NEA → flux taken directly, not computed
    row = _row(enriched, "Planet B")
    assert str(row["pl_insol_src"]) == "insol:nea"


def test_computed_flux_src_lists_all_inputs():
    # Planet with no direct insol: flux must be computed from r, teff, a.
    # Verify the source string lists all three input sources.
    from crossmatching.enrichment.param_sources.nea import NeaParamSource

    nea = NeaParamSource()
    nea._lookup = nea._build_lookup(Table({
        "pl_name":     ["No-insol b"],
        "st_teff":     [5500.0],   "st_tefferr1": [80.0],  "st_tefferr2": [-70.0],
        "st_rad":      [1.0],      "st_raderr1":  [0.05],  "st_raderr2":  [-0.04],
        "st_mass":     [1.0],      "st_masserr1": [0.05],  "st_masserr2": [-0.04],
        "st_spectype": ["G2V"],
        "pl_insol":    [0.0],      "pl_insolerr1":[np.nan],"pl_insolerr2":[np.nan],
        "sy_vmag":     [7.0],      "sy_vmagerr1": [0.02],  "sy_vmagerr2": [-0.02],
        "sy_dist":     [10.0],     "sy_disterr1": [0.5],   "sy_disterr2": [-0.4],
        "st_logg":     [4.4],      "st_loggerr1": [0.1],   "st_loggerr2": [-0.09],
        "st_met":      [np.nan],   "st_meterr1":  [np.nan],"st_meterr2":  [np.nan],
        "pl_eqt":      [np.nan],   "pl_eqterr1":  [np.nan],"pl_eqterr2":  [np.nan],
    }))
    catalog = Table({
        "exo-mercat_name": ["No-insol b"],
        "nasa_name":       ["No-insol b"],
        "main_id":         [""],
        "r":               [0.1],
        "a":               [1.0],
        "p":               [365.0],
    })
    merger = ParamFiller([nea])
    result = merger.enrich(catalog)
    src = str(result["pl_insol_src"][0])
    assert "r:" in src
    assert "teff:" in src
    assert "a:" in src


# ── Planet B: NEA only ────────────────────────────────────────────────────────

def test_planet_b_nea_teff(enriched):
    row = _row(enriched, "Planet B")
    assert float(row["st_teff"]) == pytest.approx(4800.0)


def test_planet_b_kepler_a_gives_unmasked_flux(enriched):
    # a=0 in catalog → Kepler fallback uses mass+period; pl_insol should be finite
    row = _row(enriched, "Planet B")
    assert not np.ma.is_masked(row["pl_insol"])
    assert float(row["pl_insol"]) > 0


# ── Planet C: SIMBAD only ─────────────────────────────────────────────────────

def test_planet_c_simbad_teff(enriched):
    row = _row(enriched, "Planet C")
    assert float(row["st_teff"]) == pytest.approx(3400.0)


def test_planet_c_rad_from_ms_radius(enriched):
    # mann_teff takes priority over ZAMS for teff < 4000K
    row = _row(enriched, "Planet C")
    assert float(row["st_rad"]) == pytest.approx(_mann_teff_radius(3400.0))


def test_planet_c_pl_insol_masked(enriched):
    # No a, no period, no insol → pl_insol cannot be computed
    row = _row(enriched, "Planet C")
    assert np.ma.is_masked(row["pl_insol"])


def test_planet_c_dist_from_parallax(enriched):
    row = _row(enriched, "Planet C")
    assert float(row["sy_dist"]) == pytest.approx(10.0)


# ── derived columns ───────────────────────────────────────────────────────────

def test_r_earth_conversion(enriched):
    from astropy import units as u
    factor = u.R_jup.to(u.R_earth)
    row = _row(enriched, "Planet A")
    assert float(row["r"] * R_JUP_TO_EARTH) == pytest.approx(0.1 * factor)


def test_spectral_category_non_null(enriched):
    for row in enriched:
        assert str(row["spectral_category"]) != ""


def test_planet_a_spectral_category(enriched):
    # HPIC spec "G5V" → Sun-like
    row = _row(enriched, "Planet A")
    assert str(row["spectral_category"]) == "Sun-like"


def test_planet_c_spectral_category(enriched):
    # SIMBAD sp_type "M2V" → Low-luminosity
    row = _row(enriched, "Planet C")
    assert str(row["spectral_category"]) == "Low-luminosity"


# ── Uncertainty: errors bound to winning source, kept asymmetric ──────────────

def test_planet_b_teff_err_asymmetric(enriched):
    # Planet B: NEA wins teff; err1=60, err2=50 (stored as positive magnitudes)
    row = _row(enriched, "Planet B")
    assert not np.ma.is_masked(row["st_teff_err1"])
    assert not np.ma.is_masked(row["st_teff_err2"])
    assert float(row["st_teff_err1"]) == pytest.approx(60.0)
    assert float(row["st_teff_err2"]) == pytest.approx(50.0)


def test_planet_b_rad_err_asymmetric(enriched):
    # st_raderr1=0.03, st_raderr2=0.02 — separate positive magnitudes
    row = _row(enriched, "Planet B")
    assert float(row["st_rad_err1"]) == pytest.approx(0.03)
    assert float(row["st_rad_err2"]) == pytest.approx(0.02)


def test_planet_a_teff_err_masked_because_hpic_has_no_errors(enriched):
    # Planet A: HPIC wins teff, HPIC has no error columns → both sides masked
    row = _row(enriched, "Planet A")
    assert np.ma.is_masked(row["st_teff_err1"])
    assert np.ma.is_masked(row["st_teff_err2"])


def test_planet_a_mass_err_from_nea(enriched):
    # Planet A: teff from HPIC (no err), but mass from NEA → mass_err from NEA
    # st_masserr1=0.05, st_masserr2=0.04
    row = _row(enriched, "Planet A")
    assert float(row["st_mass_err1"]) == pytest.approx(0.05)
    assert float(row["st_mass_err2"]) == pytest.approx(0.04)


def test_planet_b_lum_err1_propagated(enriched):
    # lum_err1 uses rad_err1 and teff_err1 (both push lum up)
    row = _row(enriched, "Planet B")
    assert not np.ma.is_masked(row["st_lum_err1"])
    teff   = float(row["st_teff"])
    rad    = float(row["st_rad"])
    teff_e = float(row["st_teff_err1"])
    rad_e  = float(row["st_rad_err1"])
    lum    = float(row["st_lum"])
    expected = lum * np.sqrt((2 * rad_e / rad) ** 2 + (4 * teff_e / teff) ** 2)
    assert float(row["st_lum_err1"]) == pytest.approx(expected, rel=1e-5)


def test_planet_b_lum_err2_propagated(enriched):
    # lum_err2 uses rad_err2 and teff_err2 (both push lum down)
    row = _row(enriched, "Planet B")
    assert not np.ma.is_masked(row["st_lum_err2"])
    teff   = float(row["st_teff"])
    rad    = float(row["st_rad"])
    teff_e = float(row["st_teff_err2"])
    rad_e  = float(row["st_rad_err2"])
    lum    = float(row["st_lum"])
    expected = lum * np.sqrt((2 * rad_e / rad) ** 2 + (4 * teff_e / teff) ** 2)
    assert float(row["st_lum_err2"]) == pytest.approx(expected, rel=1e-5)


def test_planet_c_rad_err_from_mann_teff_scatter(enriched):
    # Planet C: rad from mann_teff, no metallicity → fixed 13.4% intrinsic scatter
    row = _row(enriched, "Planet C")
    assert not np.ma.is_masked(row["st_rad_err1"])
    assert not np.ma.is_masked(row["st_rad_err2"])
    expected = 0.134 * _mann_teff_radius(3400.0)
    assert float(row["st_rad_err1"]) == pytest.approx(expected, rel=1e-5)
    assert float(row["st_rad_err2"]) == pytest.approx(expected, rel=1e-5)


def test_planet_b_pl_insol_err_is_direct_insol_err_asymmetric(enriched):
    # Planet B has direct insol from NEA → pl_insol_err1=insol_err1, err2=insol_err2
    # _nea_table() row 1: pl_insolerr1=0.05 (upper), pl_insolerr2=-0.04 (lower → 0.04)
    row = _row(enriched, "Planet B")
    assert not np.ma.is_masked(row["pl_insol_err1"])
    assert not np.ma.is_masked(row["pl_insol_err2"])
    assert float(row["pl_insol_err1"]) == pytest.approx(0.05)
    assert float(row["pl_insol_err2"]) == pytest.approx(0.04)


# ── temperate_mask integration ────────────────────────────────────────────────

def test_temperate_mask_central_on_enriched(enriched):
    # Planet A: pl_insol ≈ 1.2 (direct insol from NEA) → inside [0.25, 1.77]
    mask = temperate_mask(enriched["pl_insol"], enriched["pl_insol_err1"],
                          enriched["pl_insol_err2"], lower=0.25, upper=1.77)
    row_a = list(enriched["exo-mercat_name"]).index("Planet A")
    row_c = list(enriched["exo-mercat_name"]).index("Planet C")
    assert mask[row_a]       # Planet A has insol=1.2 → in range
    assert not mask[row_c]   # Planet C has masked flux → excluded


def test_temperate_mask_interval_widens_selection(enriched):
    # Check that use_interval=True can include planets just outside the central range
    flux  = enriched["pl_insol"]
    ferr1 = enriched["pl_insol_err1"]
    ferr2 = enriched["pl_insol_err2"]
    tight = temperate_mask(flux, ferr1, ferr2, lower=0.25, upper=1.77, use_interval=False)
    wide  = temperate_mask(flux, ferr1, ferr2, lower=0.25, upper=1.77, use_interval=True)
    # Interval mode can only add planets, never remove them
    assert np.all(tight <= wide)  # every True in tight is also True in wide


# ── Planet D: msini only, no direct radius ────────────────────────────────────

def test_planet_d_no_direct_radius(enriched):
    # Planet D has r=0 in catalog → treated as absent; bounds are populated instead
    row = _row(enriched, "Planet D")
    assert float(row["r"]) == 0.0
    assert not np.ma.is_masked(row["r_lower_bound"])


def test_planet_d_r_lower_bound_rocky(enriched):
    # msini=1.27 M_Earth → r_min = mass_radius_chen_kipping(1.27) in rocky range
    row = _row(enriched, "Planet D")
    assert not np.ma.is_masked(row["r_lower_bound"])
    expected_min = mass_radius_chen_kipping(1.27)
    assert float(row["r_lower_bound"]) == pytest.approx(expected_min, rel=1e-4)
    assert 0.5 < float(row["r_lower_bound"]) < 1.5


def test_planet_abc_r_lower_bound_masked(enriched):
    # Planets with direct radius have r_lower_bound/max masked
    for name in ("Planet A", "Planet B", "Planet C"):
        row = _row(enriched, name)
        assert np.ma.is_masked(row["r_lower_bound"]), f"{name} r_lower_bound should be masked"
        assert np.ma.is_masked(row["r_upper_bound"]), f"{name} r_upper_bound should be masked"


# ── rocky_mask integration ─────────────────────────────────────────────────────

def test_rocky_mask_planet_a_confirmed(enriched):
    # Planet A: r=0.1 R_Jup ≈ 1.12 R_Earth → confirmed rocky
    r = enriched["r"] * R_JUP_TO_EARTH
    rmin = enriched["r_lower_bound"]
    rmax = enriched["r_upper_bound"]
    mask = rocky_mask(r, rmin, rmax, lower=0.5, upper=1.5)
    idx = list(enriched["exo-mercat_name"]).index("Planet A")
    assert mask[idx]


def test_rocky_mask_planet_d_uncertain_rocky(enriched):
    # Planet D: no direct radius, but msini estimates put it in rocky range
    r = enriched["r"] * R_JUP_TO_EARTH
    rmin = enriched["r_lower_bound"]
    rmax = enriched["r_upper_bound"]
    idx = list(enriched["exo-mercat_name"]).index("Planet D")
    assert not rocky_mask(r, rmin, rmax, lower=0.5, upper=1.5)[idx]              # strict
    assert rocky_mask(r, rmin, rmax, lower=0.5, upper=1.5, use_interval=True)[idx]  # uncertain
