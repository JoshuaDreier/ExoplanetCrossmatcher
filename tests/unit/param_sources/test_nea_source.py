import numpy as np
import pytest
from astropy.table import Table

from crossmatching.param_sources.nea import NeaStellarParamSource


def _nea_table(*rows):
    """Minimal pscomppars-shaped table for _build_lookup tests."""
    names = [r["pl_name"] for r in rows]
    teff  = [r.get("st_teff",  0.0) for r in rows]
    rad   = [r.get("st_rad",   0.0) for r in rows]
    mass  = [r.get("st_mass",  0.0) for r in rows]
    spec  = [r.get("st_spectype", "") for r in rows]
    insol = [r.get("pl_insol", 0.0) for r in rows]
    vmag  = [r.get("sy_vmag",  0.0) for r in rows]
    dist  = [r.get("sy_dist",  0.0) for r in rows]
    logg  = [r.get("st_logg",  0.0) for r in rows]
    met   = [r.get("st_met",   np.nan) for r in rows]
    return Table({
        "pl_name":     names, "st_teff": teff,  "st_rad": rad,
        "st_mass":     mass,  "st_spectype": spec,
        "pl_insol":    insol, "sy_vmag": vmag,  "sy_dist": dist,
        "st_logg":     logg,  "st_met": met,
    })


def _loaded(src, table):
    src._lookup = src._build_lookup(table)
    return src


def _emc_row(nasa_name):
    return {"nasa_name": nasa_name}


def test_returns_correct_params():
    src = NeaStellarParamSource()
    _loaded(src, _nea_table({"pl_name": "HD 1 b", "st_teff": 5800.0, "st_rad": 1.0,
                               "st_mass": 1.0, "pl_insol": 1.1, "sy_vmag": 7.5, "sy_dist": 25.0}))
    result = src.get(_emc_row("HD 1 b"))
    assert result["teff"]  == pytest.approx(5800.0)
    assert result["insol"] == pytest.approx(1.1)
    assert result["dist"]  == pytest.approx(25.0)
    assert result["vmag"]  == pytest.approx(7.5)


def test_returns_empty_for_missing_planet():
    src = NeaStellarParamSource()
    _loaded(src, _nea_table({"pl_name": "HD 1 b", "st_teff": 5800.0}))
    assert src.get(_emc_row("Unknown e")) == {}


def test_zero_insol_not_included():
    # insol=0 means NEA doesn't have the value; should not be returned
    src = NeaStellarParamSource()
    _loaded(src, _nea_table({"pl_name": "HD 1 b", "st_teff": 5800.0, "pl_insol": 0.0}))
    result = src.get(_emc_row("HD 1 b"))
    assert "insol" not in result


def test_key_col_is_nasa_name():
    assert NeaStellarParamSource.key_col == "nasa_name"


def test_st_lum_log10_converted_to_linear():
    # pscomppars st_lum is log10(L/L_sun); negative (sub-solar) values are valid
    src = NeaStellarParamSource()
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
    assert bright["lum_err1"] == pytest.approx(10 ** 1.05 - 10.0)
    assert bright["lum_err2"] == pytest.approx(10.0 - 10 ** 0.95)
