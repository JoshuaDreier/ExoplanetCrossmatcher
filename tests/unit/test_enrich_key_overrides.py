"""Unit tests for the ``<param>_key`` override mechanism in ``ParamFiller.enrich()``.

Contract being tested
---------------------
When a caller passes e.g. ``stellar_radius_key='my_rad'``:

1. **Highest-priority source** – valid (unmasked, positive) values in
   ``table['my_rad']`` are used as-is and never overwritten by downstream
   sources.  Rows where ``table['my_rad']`` is masked / zero are filled from
   the normal source chain.

2. **Output column naming** – the merged result is stored in ``result['my_rad']``
   (not the canonical ``'st_rad'``), and the accompanying provenance/error
   columns are ``'my_rad_src'``, ``'my_rad_err1'``, ``'my_rad_err2'``.

3. **Provenance label** – rows that came from the input column carry
   ``src == 'input'``; rows filled by a downstream source carry that
   source's name.

4. **Suffix normalisation** – both ``stellar_radius_key='x'`` (via
   ``**override_keys``) and ``star_radius_key='x'`` (explicit kwarg) are
   accepted and behave identically.
"""
import numpy as np
import pytest
from tests.enrich_keys import DEFAULT_ENRICH_KEYS
from astropy.table import MaskedColumn, Table

from crossmatching.enrichment import ParamFiller
from crossmatching.enrichment.param_sources.nea import NeaParamSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nea_source(teff=5000.0, rad=1.0, mass=1.0):
    """Return a NeaParamSource pre-loaded with a single planet entry."""
    nea = NeaParamSource()
    nea._lookup = nea._build_lookup(Table({
        "pl_name":     ["Planet X"],
        "st_teff":     [teff],     "st_tefferr1": [50.0],  "st_tefferr2": [-40.0],
        "st_rad":      [rad],      "st_raderr1":  [0.05],  "st_raderr2":  [-0.04],
        "st_mass":     [mass],     "st_masserr1": [0.05],  "st_masserr2": [-0.04],
        "st_spectype": ["G2V"],
        "pl_insol":    [1.1],      "pl_insolerr1":[0.1],   "pl_insolerr2":[-0.09],
        "sy_vmag":     [7.0],      "sy_vmagerr1": [0.02],  "sy_vmagerr2": [-0.02],
        "sy_dist":     [10.0],     "sy_disterr1": [0.5],   "sy_disterr2": [-0.4],
        "st_logg":     [4.4],      "st_loggerr1": [0.1],   "st_loggerr2": [-0.09],
        "st_met":      [0.0],      "st_meterr1":  [0.05],  "st_meterr2":  [-0.05],
        "pl_eqt":      [np.nan],   "pl_eqterr1":  [np.nan],"pl_eqterr2":  [np.nan],
    }))
    return nea


def _catalog(rad_col_name: str, rad_value, mask=False):
    """Minimal catalog table with a custom radius column."""
    rad_col = MaskedColumn(
        [rad_value],
        mask=[mask],
        name=rad_col_name,
    )
    return Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        rad_col_name: rad_col,
    })


# ---------------------------------------------------------------------------
# 1. Output column naming
# ---------------------------------------------------------------------------

def test_override_output_column_uses_custom_name():
    """The merged radius should live under the caller-supplied name, not 'st_rad'."""
    nea = _nea_source(rad=0.9)
    cat = _catalog("my_rad", 1.2)
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_rad"})
    assert "my_rad" in result.colnames
    assert "st_rad" not in result.colnames


def test_override_provenance_columns_use_custom_name():
    """``my_rad_src``, ``my_rad_err1``, ``my_rad_err2`` should be present."""
    nea = _nea_source(rad=0.9)
    cat = _catalog("my_rad", 1.2)
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_rad"})
    assert "my_rad_src"  in result.colnames
    assert "my_rad_err1" in result.colnames
    assert "my_rad_err2" in result.colnames


# ---------------------------------------------------------------------------
# 2. Input is highest-priority source
# ---------------------------------------------------------------------------

def test_override_input_wins_over_nea():
    """A valid input value must not be overwritten by the NEA source."""
    nea = _nea_source(rad=0.9)          # NEA has rad=0.9
    cat = _catalog("my_rad", 1.2)       # input has rad=1.2  ← should win
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_rad"})
    assert float(result["my_rad"][0]) == pytest.approx(1.2)


def test_override_input_src_is_input():
    """When the input column supplies the value, src must be 'input'."""
    nea = _nea_source(rad=0.9)
    cat = _catalog("my_rad", 1.2)
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_rad"})
    assert str(result["my_rad_src"][0]) == "input"


# ---------------------------------------------------------------------------
# 3. Masked / absent input rows fall back to downstream sources
# ---------------------------------------------------------------------------

def test_override_masked_input_falls_back_to_nea():
    """A masked input cell should be filled by the next source (NEA)."""
    nea = _nea_source(rad=0.9)
    cat = _catalog("my_rad", 0.0, mask=True)   # masked → should fall back
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_rad"})
    assert float(result["my_rad"][0]) == pytest.approx(0.9)


def test_override_masked_input_src_is_nea():
    """When fallback from NEA fills the value, src must be 'nea'."""
    nea = _nea_source(rad=0.9)
    cat = _catalog("my_rad", 0.0, mask=True)
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_rad"})
    assert str(result["my_rad_src"][0]) == "nea"


def test_override_absent_column_falls_back_to_nea():
    """If the override column is entirely absent from the table, NEA still fills."""
    nea = _nea_source(rad=0.9)
    cat = Table({"nasa_name": ["Planet X"], "main_id": [""]})
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "missing_col"})
    assert float(result["missing_col"][0]) == pytest.approx(0.9)
    assert str(result["missing_col_src"][0]) == "nea"


# ---------------------------------------------------------------------------
# 4. Multiple overrides can coexist
# ---------------------------------------------------------------------------

def test_multiple_overrides_independent():
    """stellar_radius_key and stellar_teff_key can both be active at once."""
    nea = _nea_source(rad=0.9, teff=5000.0)
    cat = Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        "my_rad":    MaskedColumn([1.2], mask=[False]),
        "my_teff":   MaskedColumn([5800.0], mask=[False]),
    })
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS,
                                        "star_radius_key": "my_rad",
                                        "star_effective_temperature_key": "my_teff"})
    assert "my_rad"  in result.colnames and "st_rad"  not in result.colnames
    assert "my_teff" in result.colnames and "st_teff" not in result.colnames
    assert float(result["my_rad"][0])  == pytest.approx(1.2)
    assert float(result["my_teff"][0]) == pytest.approx(5800.0)
    assert str(result["my_rad_src"][0])  == "input"
    assert str(result["my_teff_src"][0]) == "input"


# ---------------------------------------------------------------------------
# 5. Explicit keyword arg (star_radius_key=) and **override_keys form are equivalent
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 6. Non-overridden params still appear under their canonical names
# ---------------------------------------------------------------------------

def test_non_overridden_params_keep_canonical_names(tmp_path):
    """Params without an override should still appear as 'st_mass', 'sy_dist', etc."""
    nea = _nea_source()
    cat = Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        "my_rad":    MaskedColumn([1.2], mask=[False]),
    })
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_rad"})
    # mass was NOT overridden → canonical name
    assert "st_mass" in result.colnames
    assert "sy_dist" in result.colnames
    # radius WAS overridden → custom name only
    assert "my_rad" in result.colnames
    assert "st_rad" not in result.colnames


# ---------------------------------------------------------------------------
# 7. pl_insol overrides and aliases
# ---------------------------------------------------------------------------

def test_pl_insol_is_present_by_default():
    """By default, pl_insol is present and correctly populated."""
    nea = _nea_source(teff=5778.0, rad=1.0, mass=1.0)
    cat = Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        "a":         [1.0],  # semi-major axis so computed flux is also defined if needed
    })
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, **DEFAULT_ENRICH_KEYS})
    
    assert "pl_insol" in result.colnames
    assert float(result["pl_insol"][0]) == pytest.approx(1.1)
    assert str(result["pl_insol_src"][0]) == "nea"
    assert float(result["pl_insol_err1"][0]) == pytest.approx(0.1)


def test_flux_rel_key_as_override_key_behaves_as_pl_insol_override():
    """flux_rel_key in override_keys behaves as an alias for overriding pl_insol."""
    nea = _nea_source(teff=5778.0, rad=1.0, mass=1.0)
    cat = Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        "my_flux":   MaskedColumn([1.5], mask=[False]),
    })
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "planet_flux_key": "my_flux"})
    
    assert "my_flux" in result.colnames
    assert "my_flux_src" in result.colnames
    assert "my_flux_err1" in result.colnames
    assert "pl_insol" not in result.colnames
    
    assert float(result["my_flux"][0]) == pytest.approx(1.5)
    assert str(result["my_flux_src"][0]) == "input"


def test_star_effective_temperature_key_override():
    """star_effective_temperature_key overrides st_teff to the custom column."""
    nea = _nea_source(teff=5778.0)
    cat = Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        "my_teff":   MaskedColumn([5800.0], mask=[False]),
    })
    result = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_effective_temperature_key": "my_teff"})
    
    assert "my_teff" in result.colnames
    assert "my_teff_src" in result.colnames
    assert "my_teff_err1" in result.colnames
    assert "st_teff" not in result.colnames
    
    assert float(result["my_teff"][0]) == pytest.approx(5800.0)
    assert str(result["my_teff_src"][0]) == "input"


def test_unified_folding_of_all_key_params():
    """Verify that key parameters prioritize the input column, write 
    outputs to the same column name, and preserve valid values.
    """
    nea = _nea_source(teff=5778.0, rad=1.0)
    cat = Table({
        "nasa_name": ["Planet X"],
        "main_id":   [""],
        "my_radius": MaskedColumn([1.5], mask=[False]),
    })
    
    # Test using explicit kwarg
    result1 = ParamFiller([nea]).enrich(cat, **{**DEFAULT_ENRICH_KEYS, "star_radius_key": "my_radius"})
    assert "my_radius" in result1.colnames
    assert "my_radius_src" in result1.colnames
    assert "st_rad" not in result1.colnames
    assert float(result1["my_radius"][0]) == pytest.approx(1.5)  # priority was input (1.5), not nea source (1.0)
