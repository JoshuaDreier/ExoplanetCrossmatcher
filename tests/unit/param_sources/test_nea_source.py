import numpy as np
import pytest
from astropy.table import Table

from crossmatching.enrichment.param_sources.nea import NeaParamSource


def _nea_table(*rows):
    """Minimal pscomppars-shaped table for _build_lookup tests."""

    return Table({
        "pl_name":     [r["pl_name"] for r in rows], 
        "st_spectype": [r.get("st_spectype", "") for r in rows],
        **{key: [r.get(key, 0.0) for r in rows] for key in [
                "st_teff", "st_rad", "st_mass", "st_lum", "st_spectype", "pl_insol", "pl_eqt",
                "pl_massj", "pl_msinij", "sy_vmag", "sy_kmag", "sy_dist", "st_logg", "st_met"
        ]} 
    })


def _loaded(src, table):
    src._lookup = src._build_lookup(table)
    return src


def _emc_row(nasa_name):
    return {"nasa_name": nasa_name}


def test_returns_correct_params():
    src = NeaParamSource()
    _loaded(src, _nea_table({
        "pl_name": "HD 1 b", 
        "st_teff": 5800.0,
        "st_rad": 1.0,
        "st_mass": 1.0,
        "st_lum": 0.9,
        "st_logg": 4.0,
        "st_met": 0.1,
        "sy_dist": 25.0,
        "sy_vmag": 7.5,
        "sy_kmag": 12,
        "pl_insol": 1.1,
        "pl_massj": 2.0,
        "pl_msinij": 3.0,
        "pl_eqt": 300,
        "st_spectype": "G0V",
    }))
    result = src.get(_emc_row("HD 1 b"))
    assert result["teff"]  == pytest.approx(5800.0)
    assert result['rad'] == pytest.approx(1.0)
    assert result['mass'] == pytest.approx(1.0)
    assert result['lum'] # conversion in other testcase
    assert result['logg'] == pytest.approx(4.0)
    assert result['met'] == pytest.approx(0.1)
    assert result["dist"]  == pytest.approx(25.0)
    assert result["vmag"]  == pytest.approx(7.5)
    assert result['kmag'] == pytest.approx(12)
    assert result["insol"] == pytest.approx(1.1)
    assert result["pl_mass"] == pytest.approx(2.0)
    assert result['msini'] == pytest.approx(3.0)
    assert result['pl_eqt'] == pytest.approx(300)
    assert result['spec'] == "G0V"

def test_returns_empty_for_missing_planet():
    src = NeaParamSource()
    _loaded(src, _nea_table({"pl_name": "HD 1 b", "st_teff": 5800.0}))
    assert src.get(_emc_row("Unknown e")) == {}


def test_zero_insol_not_included():
    # insol=0 means NEA doesn't have the value; should not be returned
    src = NeaParamSource()
    _loaded(src, _nea_table({"pl_name": "HD 1 b", "st_teff": 5800.0, "pl_insol": 0.0}))
    result = src.get(_emc_row("HD 1 b"))
    assert "insol" not in result


def test_key_col_is_nasa_name():
    assert NeaParamSource.key_col == "nasa_name"


def test_st_lum_log10_converted_to_linear():
    # pscomppars st_lum is log10(L/L_sun); negative (sub-solar) values are valid
    src = NeaParamSource()
    src._lookup = src._build_lookup(Table({
        "pl_name":    ["Dim b",  "Bright b"],
        "st_lum":     [-2.8,     1.0],
        "st_lumerr1": [np.nan,   0.05],
        "st_lumerr2": [np.nan,  -0.05],
    }))
    dim = src.get(_emc_row("Dim b"))
    assert dim["lum"] == pytest.approx(10 ** -2.8)
    bright = src.get(_emc_row("Bright b"))
    assert bright["lum"] == pytest.approx(10.0)
    # dex errors are converted through the exponential, each side separately
    assert bright["lumerr1"] == pytest.approx(10 ** 1.05 - 10.0)
    assert bright["lumerr2"] == pytest.approx(10.0 - 10 ** 0.95)
