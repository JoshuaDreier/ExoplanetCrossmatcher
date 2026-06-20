from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table

from crossmatching.enrichment.param_sources.epic import EpicParamSource


def _epic_table(*rows):
    """Minimal k2pandc-shaped table for _build_lookup tests."""
    
    qty_list = ["st_teff", "st_rad", "st_mass", "st_lum", "st_spectype", "pl_insol", "pl_eqt",
                "pl_massj", "pl_msinij", "sy_vmag", "sy_kmag", "sy_dist", "st_logg", "st_met"]
    qty_list_errs = []
    for i in qty_list:
        qty_list_errs.append(i+"err1")
        qty_list_errs.append(i+"err2")
    
    return Table({
        "pl_name":     [r["pl_name"] for r in rows], 
        "st_spectype": [r.get("st_spectype", "") for r in rows],
        **{key: [r.get(key, np.nan) for r in rows] for key in qty_list + qty_list_errs} 
    })


def _loaded(src, table):
    src._lookup = src._build_lookup(table)
    return src


def _emc_row(epic_name):
    return {"epic_name": epic_name}


def test_returns_standard_params():
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-1 b", "st_teff": 4800.0, "st_rad": 0.7,
                               "st_mass": 0.8, "pl_insol": 0.9, "sy_vmag": 10.1,
                               "sy_dist": 80.0, "st_spectype": "K5V"}))
    result = src.get(_emc_row("K2-1 b"))
    assert result["teff"]  == pytest.approx(4800.0)
    assert result["rad"]   == pytest.approx(0.7)
    assert result["mass"]  == pytest.approx(0.8)
    assert result["insol"] == pytest.approx(0.9)
    assert result["vmag"]  == pytest.approx(10.1)
    assert result["dist"]  == pytest.approx(80.0)
    assert result["spec"]  == "K5V"


def test_returns_logg_met_pl_eqt():
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-2 b", "st_teff": 5000.0,
                               "st_logg": 4.5, "st_met": -0.2, "pl_eqt": 350.0}))
    result = src.get(_emc_row("K2-2 b"))
    assert result["logg"]   == pytest.approx(4.5)
    assert result["met"]    == pytest.approx(-0.2)
    assert result["pl_eqt"] == pytest.approx(350.0)


def test_solar_metallicity_zero_is_included():
    # met=0.0 means [Fe/H]=0 (solar), not "missing data" — must be included
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-3 b", "st_teff": 5778.0, "st_met": 0.0}))
    result = src.get(_emc_row("K2-3 b"))
    assert "met" in result
    assert result["met"] == pytest.approx(0.0)


def test_nan_metallicity_not_included():
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-4 b", "st_teff": 5000.0}))  # st_met defaults to nan
    result = src.get(_emc_row("K2-4 b"))
    assert "met" not in result


def test_zero_pl_eqt_not_included():
    # pl_eqt=0 is a sentinel (require_positive=True), not a real temperature
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-5 b", "st_teff": 4500.0, "pl_eqt": 0.0}))
    result = src.get(_emc_row("K2-5 b"))
    assert "pl_eqt" not in result


def test_returns_empty_for_unknown_name():
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-1 b", "st_teff": 4800.0}))
    assert src.get(_emc_row("K2-999 z")) == {}


def test_first_occurrence_wins_on_duplicate():
    src = EpicParamSource()
    _loaded(src, _epic_table(
        {"pl_name": "K2-6 b", "st_teff": 4000.0},
        {"pl_name": "K2-6 b", "st_teff": 9999.0},
    ))
    assert src.get(_emc_row("K2-6 b"))["teff"] == pytest.approx(4000.0)


@patch("crossmatching.enrichment.param_sources.epic.pyvo.dal.TAPService")
def test_download_queries_k2pandc(MockTAP):
    MockTAP.return_value.run_sync.return_value.to_table.return_value = _epic_table()
    src = EpicParamSource()
    src.download()
    query = MockTAP.return_value.run_sync.call_args[0][0]
    assert "k2pandc" in query.lower()


def test_asymmetric_errors_returned_separately():
    # err1=+100 (upper), err2=-80 (lower) → stored as positive magnitudes
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-7 b", "st_teff": 5000.0,
                               "st_tefferr1": 100.0, "st_tefferr2": -80.0}))
    result = src.get(_emc_row("K2-7 b"))
    assert result["tefferr1"] == pytest.approx(100.0)
    assert result["tefferr2"] == pytest.approx(80.0)


def test_no_err_when_both_absent():
    # No error columns provided → no *err1/*err2 keys
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-8 b", "st_teff": 5000.0}))
    result = src.get(_emc_row("K2-8 b"))
    assert "tefferr1" not in result
    assert "tefferr2" not in result


def test_rad_err_returned_asymmetric():
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-9 b", "st_teff": 5000.0, "st_rad": 1.0,
                               "st_raderr1": 0.05, "st_raderr2": -0.04}))
    result = src.get(_emc_row("K2-9 b"))
    assert result["raderr1"] == pytest.approx(0.05)
    assert result["raderr2"] == pytest.approx(0.04)


def test_met_err_zero_base_still_returns_err():
    # met=0.0 is solar, should still get its error
    src = EpicParamSource()
    _loaded(src, _epic_table({"pl_name": "K2-10 b", "st_teff": 5778.0, "st_met": 0.0,
                               "st_meterr1": 0.05, "st_meterr2": -0.05}))
    result = src.get(_emc_row("K2-10 b"))
    assert "met" in result
    assert result["meterr1"] == pytest.approx(0.05)
    assert result["meterr1"] == pytest.approx(0.05)
