import numpy as np
import pytest
from tests.enrich_keys import DEFAULT_ENRICH_KEYS
from astropy.table import MaskedColumn, Table

from crossmatching.enrichment import ParamFiller
from crossmatching.enrichment.param_sources.nea import NeaParamSource

def _nea_source(teff=5778.0, rad=None, mass=None, logg=None):
    """Return a NeaParamSource pre-loaded with a single planet entry."""
    nea = NeaParamSource()
    nea._lookup = nea._build_lookup(Table({
        "pl_name":     ["Planet X"],
        "st_teff":     [teff],
        "st_tefferr1": [50.0],
        "st_tefferr2": [-40.0],
        "st_rad":      [rad if rad is not None else np.nan],
        "st_raderr1":  [0.05 if rad is not None else np.nan],
        "st_raderr2":  [-0.04 if rad is not None else np.nan],
        "st_mass":     [mass if mass is not None else np.nan],
        "st_masserr1": [0.05 if mass is not None else np.nan],
        "st_masserr2": [-0.04 if mass is not None else np.nan],
        "st_spectype": ["G2V"],
        "pl_insol":    [np.nan],
        "pl_insolerr1":[np.nan],
        "pl_insolerr2":[np.nan],
        "sy_vmag":     [7.0],      "sy_vmagerr1": [0.02],  "sy_vmagerr2": [-0.02],
        "sy_dist":     [10.0],     "sy_disterr1": [0.5],   "sy_disterr2": [-0.4],
        "st_logg":     [logg if logg is not None else np.nan],
        "st_loggerr1": [0.05 if logg is not None else np.nan],
        "st_loggerr2": [-0.05 if logg is not None else np.nan],
        "st_met":      [0.0],      "st_meterr1": [0.05],   "st_meterr2": [-0.05],
        "st_lum":      [np.nan],   "st_lumerr1": [np.nan], "st_lumerr2": [np.nan],
    }))
    return nea

def test_disable_calculations_skips_derivations():
    """Verify that disable_calculations=True prevents deriving mass and adding derived columns."""
    nea = _nea_source(rad=1.0, logg=4.43) # mass is omitted, would normally be derived
    
    cat = Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        "msini":     [1.0],  # Normally triggers r_lower_bound computation
    })

    # Act - Normal run (Calculations Enabled)
    merger = ParamFiller([nea])
    result_enabled = merger.enrich(cat, **DEFAULT_ENRICH_KEYS)[0]
    
    # Assert normal behavior
    assert "r_lower_bound" in result_enabled.colnames
    assert not result_enabled["st_mass"].mask[0] # Mass was derived

    # Act - Disabled calculations
    result_disabled = merger.enrich(cat, disable_calculations=True, **DEFAULT_ENRICH_KEYS)[0]
    
    # Assert disabled behavior
    assert "r_lower_bound" not in result_disabled.colnames
    assert "a_src" not in result_disabled.colnames
    assert "spectral_category" not in result_disabled.colnames
    assert result_disabled["st_mass"].mask[0] # Mass was NOT derived, should be masked/empty
    assert "st_mass" in result_disabled.colnames # Column still exists because of _merge_values

