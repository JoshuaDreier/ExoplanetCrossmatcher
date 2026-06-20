import numpy as np
import pytest
from astropy.table import Table

from crossmatching.enrichment.param_sources.eu import EuParamSource


def _eu_table(*rows):
    """Minimal eu.core-shaped table for _build_lookup tests."""

    return Table({
        "name":     [r["name"] for r in rows], 
        "star_spectype": [r.get("star_spectype", "") for r in rows],
        **{key: [r.get(key, 0.0) for r in rows] for key in [
            'star_teff', 'star_radius', 'star_mass', 'star_distance', 'star_metallicity', 'mag_v', 
            'mag_k', 'temp_calculated', 'radius', 'mass', 'mass_sin_i', 'star_spec_type'
        ]} 
    })


def _loaded(src, table):
    src._lookup = src._build_lookup(table)
    return src


def _emc_row(eu_name):
    return {"eu_name": eu_name}


def test_returns_correct_params():
    src = EuParamSource()
    _loaded(src, _eu_table({
        "name": "HD 1 b", 
        "star_teff": 5800.0,
        "star_radius": 1.0,
        "star_mass": 1.0,
        "star_distance": 25,
        "star_metallicity": 0.1,
        "mag_v": 7.5,
        "mag_k": 12.0,
        "mass": 2.0,
        "mass_sin_i": 3.0,
        "temp_calculated": 300,
        "star_spec_type": "G0V",
    }))
    result = src.get(_emc_row("HD 1 b"))
    assert result["teff"]  == pytest.approx(5800.0)
    assert result['rad'] == pytest.approx(1.0)
    assert result['mass'] == pytest.approx(1.0)
    assert result["dist"]  == pytest.approx(25.0)
    assert result['met'] == pytest.approx(0.1)
    assert result["vmag"]  == pytest.approx(7.5)
    assert result['kmag'] == pytest.approx(12)
    assert result["pl_mass"] == pytest.approx(2.0)
    assert result['msini'] == pytest.approx(3.0)
    assert result['pl_eqt'] == pytest.approx(300)
    assert result['spec'] == "G0V"

def test_returns_empty_for_missing_planet():
    src = EuParamSource()
    _loaded(src, _eu_table({"name": "HD 1 b", "star_teff": 5800.0}))
    assert src.get(_emc_row("Unknown e")) == {}


def test_key_col_is_eu_name():
    assert EuParamSource.key_col == "eu_name"

