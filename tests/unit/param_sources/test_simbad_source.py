from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table

from crossmatching.param_sources.simbad import SimbadStellarParamSource


def _simbad_result(*rows):
    """Fake SIMBAD TAP result matching the query in SimbadStellarParamSource.download()."""
    mids    = [r["main_id"]              for r in rows]
    teff    = [r.get("teff",      np.nan) for r in rows]
    spec    = [r.get("sp_type",       "") for r in rows]
    plx     = [r.get("plx_value",  0.0)  for r in rows]
    plx_err = [r.get("plx_err",   np.nan) for r in rows]
    vmag    = [r.get("vmag",      np.nan) for r in rows]
    return Table({"main_id": mids, "teff": teff, "sp_type": spec,
                  "plx_value": plx, "plx_err": plx_err, "vmag": vmag})


def _loaded(src, table):
    src._lookup = src._build_lookup(table)
    return src


def _emc_row(mid):
    return {"main_id": mid}


def test_returns_teff_and_spec():
    src = SimbadStellarParamSource()
    _loaded(src, _simbad_result({"main_id": "Proxima", "teff": 3000.0, "sp_type": "M5.5V"}))
    result = src.get(_emc_row("Proxima"))
    assert result["teff"] == pytest.approx(3000.0)
    assert result["spec"] == "M5.5V"



def test_computes_dist_from_parallax():
    src = SimbadStellarParamSource()
    _loaded(src, _simbad_result({"main_id": "Proxima", "teff": 3000.0, "plx_value": 769.0}))
    result = src.get(_emc_row("Proxima"))
    assert result["dist"] == pytest.approx(1000.0 / 769.0)


def test_returns_empty_for_missing_main_id():
    src = SimbadStellarParamSource()
    _loaded(src, _simbad_result({"main_id": "Proxima", "teff": 3000.0}))
    assert src.get(_emc_row("Unknown star")) == {}


def test_zero_plx_not_converted_to_dist():
    # parallax=0 → infinite distance → should not be included
    src = SimbadStellarParamSource()
    _loaded(src, _simbad_result({"main_id": "Star", "teff": 4000.0, "plx_value": 0.0}))
    result = src.get(_emc_row("Star"))
    assert "dist" not in result


@patch("crossmatching.param_sources.simbad.pyvo.dal.TAPService")
def test_download_uploads_key_list(MockTAP):
    MockTAP.return_value.run_sync.return_value.to_table.return_value = _simbad_result()
    src = SimbadStellarParamSource()
    src.download(["* Proxima Cen", "GJ 447"])
    call_kwargs = MockTAP.return_value.run_sync.call_args
    uploaded = call_kwargs[1]["uploads"]["ids"]
    assert list(uploaded["main_id"]) == ["* Proxima Cen", "GJ 447"]


def test_dist_err_from_parallax_error():
    # d = 1000/plx → σ_d = 1000 * σ_plx / plx²; symmetric → err1 == err2
    # plx=200 → d=5pc, plx_err=2 → dist_err = 1000*2/200² = 0.05 pc
    src = SimbadStellarParamSource()
    _loaded(src, _simbad_result({"main_id": "Star", "teff": 4000.0,
                                  "plx_value": 200.0, "plx_err": 2.0}))
    result = src.get(_emc_row("Star"))
    assert result["dist"] == pytest.approx(5.0)
    assert result["dist_err1"] == pytest.approx(0.05)
    assert result["dist_err2"] == pytest.approx(0.05)


def test_no_dist_err_when_plx_err_absent():
    src = SimbadStellarParamSource()
    _loaded(src, _simbad_result({"main_id": "Star2", "teff": 4000.0, "plx_value": 100.0}))
    result = src.get(_emc_row("Star2"))
    assert "dist" in result
    assert "dist_err1" not in result
    assert "dist_err2" not in result


def test_download_includes_plx_err_in_query():
    with patch("crossmatching.param_sources.simbad.pyvo.dal.TAPService") as MockTAP:
        MockTAP.return_value.run_sync.return_value.to_table.return_value = _simbad_result()
        src = SimbadStellarParamSource()
        src.download(["Star"])
        query = MockTAP.return_value.run_sync.call_args[0][0]
        assert "plx_err" in query
