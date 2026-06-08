from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table

from crossmatching.param_sources.toi import ToiStellarParamSource


def _toi_table(*rows):
    """Minimal toi-shaped table for _build_lookup tests."""
    return Table({
        "toidisplay":  [r["toidisplay"]               for r in rows],
        "st_teff":     [r.get("st_teff",    0.0)      for r in rows],
        "st_tefferr1": [r.get("st_tefferr1", np.nan)  for r in rows],
        "st_tefferr2": [r.get("st_tefferr2", np.nan)  for r in rows],
        "st_rad":      [r.get("st_rad",     0.0)      for r in rows],
        "st_raderr1":  [r.get("st_raderr1", np.nan)   for r in rows],
        "st_raderr2":  [r.get("st_raderr2", np.nan)   for r in rows],
        "st_dist":     [r.get("st_dist",    0.0)      for r in rows],
        "st_disterr1": [r.get("st_disterr1", np.nan)  for r in rows],
        "st_disterr2": [r.get("st_disterr2", np.nan)  for r in rows],
        "st_logg":     [r.get("st_logg",    0.0)      for r in rows],
        "st_loggerr1": [r.get("st_loggerr1", np.nan)  for r in rows],
        "st_loggerr2": [r.get("st_loggerr2", np.nan)  for r in rows],
        "pl_insol":    [r.get("pl_insol",   0.0)      for r in rows],
        "pl_insolerr1":[r.get("pl_insolerr1", np.nan) for r in rows],
        "pl_insolerr2":[r.get("pl_insolerr2", np.nan) for r in rows],
        "pl_eqt":      [r.get("pl_eqt",    0.0)      for r in rows],
        "pl_eqterr1":  [r.get("pl_eqterr1", np.nan)  for r in rows],
        "pl_eqterr2":  [r.get("pl_eqterr2", np.nan)  for r in rows],
    })


def _loaded(src, table):
    src._lookup = src._build_lookup(table)
    return src


def _emc_row(toi_name):
    return {"toi_name": toi_name}


def test_returns_available_params():
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-700.01", "st_teff": 3480.0, "st_rad": 0.42,
                              "st_dist": 31.1, "st_logg": 4.7, "pl_insol": 0.9, "pl_eqt": 268.0}))
    result = src.get(_emc_row("TOI-700.01"))
    assert result["teff"]   == pytest.approx(3480.0)
    assert result["rad"]    == pytest.approx(0.42)
    assert result["dist"]   == pytest.approx(31.1)
    assert result["logg"]   == pytest.approx(4.7)
    assert result["insol"]  == pytest.approx(0.9)
    assert result["pl_eqt"] == pytest.approx(268.0)


def test_dist_comes_from_st_dist_not_sy_dist():
    # TOI uses st_dist; ensure the mapper reads that column
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-100.01", "st_teff": 5000.0, "st_dist": 55.0}))
    result = src.get(_emc_row("TOI-100.01"))
    assert result["dist"] == pytest.approx(55.0)


def test_no_spec_or_vmag_provided():
    # TOI has neither spectral type nor V-band magnitude
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-200.01", "st_teff": 5200.0}))
    result = src.get(_emc_row("TOI-200.01"))
    assert "spec" not in result
    assert "vmag" not in result


def test_zero_logg_not_included():
    # logg=0 is a sentinel (require_positive=True)
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-300.01", "st_teff": 4800.0, "st_logg": 0.0}))
    result = src.get(_emc_row("TOI-300.01"))
    assert "logg" not in result


def test_zero_pl_eqt_not_included():
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-400.01", "st_teff": 4500.0, "pl_eqt": 0.0}))
    result = src.get(_emc_row("TOI-400.01"))
    assert "pl_eqt" not in result


def test_returns_empty_for_unknown_toi():
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-700.01", "st_teff": 3480.0}))
    assert src.get(_emc_row("TOI-9999.99")) == {}


@patch("crossmatching.param_sources.toi.pyvo.dal.TAPService")
def test_download_queries_toi_table(MockTAP):
    MockTAP.return_value.run_sync.return_value.to_table.return_value = _toi_table()
    src = ToiStellarParamSource()
    src.download()
    query = MockTAP.return_value.run_sync.call_args[0][0]
    assert "toi" in query.lower()


def test_teff_err_asymmetric():
    # err1=+120 (upper), err2=-90 (lower) → stored as positive magnitudes
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-500.01", "st_teff": 4800.0,
                              "st_tefferr1": 120.0, "st_tefferr2": -90.0}))
    result = src.get(_emc_row("TOI-500.01"))
    assert result["teff_err1"] == pytest.approx(120.0)
    assert result["teff_err2"] == pytest.approx(90.0)


def test_dist_err_uses_st_disterr_columns():
    # Verifies TOI-specific st_disterr1/2 (not sy_disterr) and asymmetric storage
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-600.01", "st_teff": 5000.0,
                              "st_dist": 50.0, "st_disterr1": 2.0, "st_disterr2": -1.5}))
    result = src.get(_emc_row("TOI-600.01"))
    assert result["dist_err1"] == pytest.approx(2.0)
    assert result["dist_err2"] == pytest.approx(1.5)


def test_no_err_when_absent():
    src = ToiStellarParamSource()
    _loaded(src, _toi_table({"toidisplay": "TOI-700.02", "st_teff": 3480.0}))
    result = src.get(_emc_row("TOI-700.02"))
    assert "teff_err1" not in result
    assert "teff_err2" not in result
