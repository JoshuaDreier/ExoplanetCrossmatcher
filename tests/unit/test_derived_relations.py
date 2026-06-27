import numpy as np
import pytest
from tests.enrich_keys import DEFAULT_ENRICH_KEYS
from astropy.table import MaskedColumn, Table

from crossmatching.enrichment import ParamFiller
from crossmatching.enrichment.param_sources.nea import NeaParamSource


def _nea_source(teff=5778.0, rad=None, mass=None, logg=None, insol=None, eqt=None, lum=None, spectype="G2V", teff_masked=False):
    """Return a NeaParamSource pre-loaded with a single planet entry."""
    nea = NeaParamSource()
    nea._lookup = nea._build_lookup(Table({
        "pl_name":     ["Planet X"],
        "st_teff":     [np.nan if teff_masked else teff],
        "st_tefferr1": [np.nan if teff_masked else 50.0],
        "st_tefferr2": [np.nan if teff_masked else -40.0],
        "st_rad":      [rad if rad is not None else np.nan],
        "st_raderr1":  [0.05 if rad is not None else np.nan],
        "st_raderr2":  [-0.04 if rad is not None else np.nan],
        "st_mass":     [mass if mass is not None else np.nan],
        "st_masserr1": [0.05 if mass is not None else np.nan],
        "st_masserr2": [-0.04 if mass is not None else np.nan],
        "st_spectype": [spectype],
        "pl_insol":    [insol if insol is not None else np.nan],
        "pl_insolerr1":[0.1 if insol is not None else np.nan],
        "pl_insolerr2":[-0.09 if insol is not None else np.nan],
        "sy_vmag":     [7.0],      "sy_vmagerr1": [0.02],  "sy_vmagerr2": [-0.02],
        "sy_dist":     [10.0],     "sy_disterr1": [0.5],   "sy_disterr2": [-0.4],
        "st_logg":     [logg if logg is not None else np.nan],
        "st_loggerr1": [0.1 if logg is not None else np.nan],
        "st_loggerr2": [-0.09 if logg is not None else np.nan],
        "st_met":      [0.0],      "st_meterr1":  [0.05],  "st_meterr2":  [-0.05],
        "pl_eqt":      [eqt if eqt is not None else np.nan],
        "pl_eqterr1":  [0.1 * eqt if eqt is not None else np.nan],
        "pl_eqterr2":  [-0.1 * eqt if eqt is not None else np.nan],
        "st_lum":      [np.log10(lum) if lum is not None else np.nan],
        "st_lumerr1":  [0.05 if lum is not None else np.nan],
        "st_lumerr2":  [-0.04 if lum is not None else np.nan],
    }))
    return nea


def test_rad_derived_from_logg_and_mass():
    # If st_rad is missing (masked), but st_mass and st_logg are provided, R is derived.
    # logg = 4.43797 + log10(M) - 2log10(R) -> R = 10^((4.43797 + log10(M) - logg)/2)
    # Using M = 1.0, logg = 4.43797, R should be 1.0
    nea = _nea_source(mass=1.0, logg=4.43797, rad=None)
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    
    result = ParamFiller([nea]).enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    assert "st_rad" in result.colnames
    assert float(result["st_rad"][0]) == pytest.approx(1.0, rel=1e-4)
    assert "logg_derived" in str(result["st_rad_src"][0])


def test_mass_derived_from_logg_and_rad():
    # If st_mass is missing, but st_rad and st_logg are provided, M is derived.
    # M = 10^(logg - 4.43797 + 2log10(R))
    # Using R = 1.0, logg = 4.43797, M should be 1.0
    nea = _nea_source(rad=1.0, logg=4.43797, mass=None)
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    
    result = ParamFiller([nea]).enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    assert "st_mass" in result.colnames
    assert float(result["st_mass"][0]) == pytest.approx(1.0, rel=1e-4)
    assert "logg_derived" in str(result["st_mass_src"][0])


def test_eqt_derived_from_insol():
    # If pl_eqt is missing, but pl_insol is provided, pl_eqt is derived.
    # T_eq = 254.793 * S_eff^(0.25)
    # For S_eff = 1.0, T_eq = 254.793
    nea = _nea_source(insol=1.0, eqt=None)
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    
    result = ParamFiller([nea]).enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    assert "pl_eqt" in result.colnames
    assert float(result["pl_eqt"][0]) == pytest.approx(254.793, rel=1e-4)
    assert "derived(insol:" in str(result["pl_eqt_src"][0])


def test_teff_derived_from_spectype():
    # If st_teff is missing, but st_spectype is provided, Teff is derived.
    # For G2V, spectype_to_teff("G2V") should be 5780 K.
    # The closest matched anchor range for G2 is [5660, 5860] (neighbors G1 and G5).
    # So upper error (err1) = 5860 - 5780 = 80 K, lower error (err2) = 5780 - 5660 = 120 K
    nea = _nea_source(teff_masked=True, spectype="G2V")
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    
    result = ParamFiller([nea]).enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    assert "st_teff" in result.colnames
    assert float(result["st_teff"][0]) == pytest.approx(5780.0, rel=1e-4)
    assert float(result["st_tefferr1"][0]) == pytest.approx(80.0, rel=1e-4)
    assert float(result["st_tefferr2"][0]) == pytest.approx(120.0, rel=1e-4)
    assert "spectype_derived" in str(result["st_teff_src"][0])


def test_teff_derived_from_rad_and_lum():
    # If st_teff is missing, but st_rad and st_lum are provided, Teff is derived using SB.
    # T = T_sun * (L / R^2)^0.25. Pass linear lum=1.0
    nea = _nea_source(teff_masked=True, rad=1.0, lum=1.0, spectype="")
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    
    result = ParamFiller([nea]).enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    assert "st_teff" in result.colnames
    assert float(result["st_teff"][0]) == pytest.approx(5778.0, rel=1e-4)
    assert float(result["st_tefferr1"][0]) == pytest.approx(210.7609, rel=1e-4)
    assert float(result["st_tefferr2"][0]) == pytest.approx(192.4066, rel=1e-4)
    assert "StephanBoltzmann_derived" in str(result["st_teff_src"][0])


def test_rad_derived_from_lum_and_teff():
    # If st_rad is missing, and st_teff and st_lum are provided, R is derived using SB.
    # R = sqrt(L) * (T_sun / T)^2. Pass linear lum=1.0
    nea = _nea_source(teff=5778.0, rad=None, lum=1.0)
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    
    result = ParamFiller([nea]).enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    assert "st_rad" in result.colnames
    assert float(result["st_rad"][0]) == pytest.approx(1.0, rel=1e-4)
    assert "StephanBoltzmann_derived" in str(result["st_rad_src"][0])


def test_insol_derived_from_eqt():
    # If pl_insol is missing, but pl_eqt is provided, pl_insol is derived.
    # S_eff = (T_eqt / 254.793)^4
    nea = _nea_source(insol=None, eqt=254.793)
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    
    result = ParamFiller([nea]).enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    assert "pl_insol" in result.colnames
    assert float(result["pl_insol"][0]) == pytest.approx(1.0, rel=1e-4)
    assert "derived(eqt:" in str(result["pl_insol_src"][0])

