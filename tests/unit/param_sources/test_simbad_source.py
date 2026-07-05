from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table
import astropy.units as u 
from crossmatching.enrichment.param_sources.simbad import SimbadParamSource


def _simbad_result(*rows):
    """Fake SIMBAD TAP result matching the query in SimbadParamSource.download()."""
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
    src = SimbadParamSource()
    _loaded(src, _simbad_result({"main_id": "Proxima", "teff": 3000.0, "sp_type": "M5.5V"}))
    result = src.get(_emc_row("Proxima"))
    assert result["teff"] == pytest.approx(3000.0)
    assert result["spec"] == "M5.5V"



def test_computes_dist_from_parallax():
    src = SimbadParamSource()
    _loaded(src, _simbad_result({"main_id": "Proxima", "teff": 3000.0, "plx_value": 769.0}))
    result = src.get(_emc_row("Proxima"))
    assert result["dist"] == pytest.approx(1000.0 / 769.0)


def test_returns_empty_for_missing_main_id():
    src = SimbadParamSource()
    _loaded(src, _simbad_result({"main_id": "Proxima", "teff": 3000.0}))
    assert src.get(_emc_row("Unknown star")) == {}


def test_zero_plx_not_converted_to_dist():
    # parallax=0 → infinite distance → should not be included
    src = SimbadParamSource()
    _loaded(src, _simbad_result({"main_id": "Star", "teff": 4000.0, "plx_value": 0.0}))
    result = src.get(_emc_row("Star"))
    assert "dist" not in result


@patch("crossmatching.enrichment.param_sources.simbad.pyvo.dal.TAPService")
def test_download_uploads_key_list(MockTAP):
    MockTAP.return_value.run_sync.return_value.to_table.return_value = _simbad_result()
    src = SimbadParamSource()
    src.download(["* Proxima Cen", "GJ 447"])
    call_kwargs = MockTAP.return_value.run_sync.call_args
    uploaded = call_kwargs[1]["uploads"]["ids"]
    assert list(uploaded["main_id"]) == ["* Proxima Cen", "GJ 447"]


def test_dist_err_from_parallax_error():
    # d = 1000/plx → σ_d = 1000 * σ_plx / plx²; symmetric → err1 == err2
    # plx=200 → d=5pc, plx_err=2 → dist_err = 1000*2/200² = 0.05 pc
    src = SimbadParamSource()
    _loaded(src, _simbad_result({"main_id": "Star", "teff": 4000.0,
                                  "plx_value": 200.0, "plx_err": 2.0}))
    result = src.get(_emc_row("Star"))
    assert result["dist"] == pytest.approx(5.0)
    assert result["disterr1"] == pytest.approx(0.05)
    assert result["disterr1"] == pytest.approx(0.05)


def test_no_dist_err_when_plx_err_absent():
    src = SimbadParamSource()
    _loaded(src, _simbad_result({"main_id": "Star2", "teff": 4000.0, "plx_value": 100.0}))
    result = src.get(_emc_row("Star2"))
    assert "dist" in result
    assert "disterr1" not in result
    assert "disterr1" not in result


def test_download_includes_plx_err_in_query():
    with patch("crossmatching.enrichment.param_sources.simbad.pyvo.dal.TAPService") as MockTAP:
        MockTAP.return_value.run_sync.return_value.to_table.return_value = _simbad_result()
        src = SimbadParamSource()
        src.download(["Star"])
        # call_args_list[0] is the main stellar params query; [1] is the diameter query
        main_query = MockTAP.return_value.run_sync.call_args_list[0][0][0]
        assert "plx_err" in main_query


def _with_diameters(main_table, diam_rows):
    """Attach a diameter Table to main_table.meta and call preprocess()."""
    main_table.meta["_diameter_rows"] = diam_rows
    src = SimbadParamSource()
    return src.preprocess(main_table)


def _diam_table(rows):
    """Build a fake mesDiameter result table."""
    return Table({
        "main_id":  [r["main_id"]             for r in rows],
        "diameter": [r.get("diameter", np.nan) for r in rows],
        "error":    [r.get("error",    np.nan) for r in rows],
        "unit":     [r.get("unit",     "km  ") for r in rows],
    })


def test_rad_from_km_diameter():
    # Sun: diameter = 2 * R_sun in km → rad should be exactly 1.0 R_sun
    d_sun_km = 2.0 * (1.0 * u.R_sun).to(u.km).value
    main = _simbad_result({"main_id": "Sol", "teff": 5778.0, "plx_value": 0.0})
    diam = _diam_table([{"main_id": "Sol", "diameter": d_sun_km, "unit": "km  "}])
    out = _with_diameters(main, diam)
    src = SimbadParamSource()
    src._lookup = src._build_lookup(out)
    result = src.get({"main_id": "Sol"})
    assert "rad" in result
    assert result["rad"] == pytest.approx(1.0, rel=1e-4)


def test_rad_from_km_diameter_error():
    # Error on the diameter should propagate to radius as half that amount
    # because radius = diameter / 2.
    e_km = 0.1 * (1.0 * u.R_sun).to(u.km).value
    d_sun_km = 2.0 * (1.0 * u.R_sun).to(u.km).value
    main = _simbad_result({"main_id": "Sol", "teff": 5778.0, "plx_value": 0.0})
    diam = _diam_table([{"main_id": "Sol", "diameter": d_sun_km, "error": e_km, "unit": "km  "}])
    out = _with_diameters(main, diam)
    src = SimbadParamSource()
    src._lookup = src._build_lookup(out)
    result = src.get({"main_id": "Sol"})
    assert result["raderr1"] == pytest.approx(0.05, rel=1e-4)
    assert result["raderr1"] == result["raderr2"]


def test_rad_from_mas_diameter_with_distance():
    # Star at 10 pc with an angular diameter consistent with 2 R_sun → rad ≈ 2.0.
    d_pc = 10.0
    r_sun_expected = 2.0
    r_km = r_sun_expected * (1.0 * u.R_sun).to(u.km).value
    theta_rad = r_km / (d_pc * (1.0 * u.pc).to(u.km).value)
    theta_mas = 2.0 * theta_rad * (648_000.0 / np.pi) * 1e3
    plx = 1000.0 / d_pc
    main = _simbad_result({"main_id": "Star", "teff": 4000.0, "plx_value": plx})
    diam = _diam_table([{"main_id": "Star", "diameter": theta_mas, "unit": "mas "}])
    out = _with_diameters(main, diam)
    src = SimbadParamSource()
    src._lookup = src._build_lookup(out)
    result = src.get({"main_id": "Star"})
    assert "rad" in result
    assert result["rad"] == pytest.approx(r_sun_expected, rel=1e-4)


def test_rad_from_mas_diameter_error_with_distance():
    # A 10% diameter error should propagate to a 10% radius error.
    d_pc = 10.0
    r_sun_expected = 2.0
    r_km = r_sun_expected * (1.0 * u.R_sun).to(u.km).value
    theta_rad = r_km / (d_pc * (1.0 * u.pc).to(u.km).value)
    theta_mas = 2.0 * theta_rad * (648_000.0 / np.pi) * 1e3
    theta_err_mas = 0.1 * theta_mas
    plx = 1000.0 / d_pc
    main = _simbad_result({"main_id": "Star", "teff": 4000.0, "plx_value": plx})
    diam = _diam_table([{"main_id": "Star", "diameter": theta_mas, "error": theta_err_mas, "unit": "mas "}])
    out = _with_diameters(main, diam)
    src = SimbadParamSource()
    src._lookup = src._build_lookup(out)
    result = src.get({"main_id": "Star"})
    assert result["raderr1"] == pytest.approx(0.2, rel=1e-4)
    assert result["raderr1"] == result["raderr2"]


def test_rad_km_preferred_over_mas():
    # When both km and mas entries exist for the same star, km wins
    d_pc = 10.0
    plx = 1000.0 / d_pc
    km_expected = 1.0  # 1 R_sun from km measurement
    d_km = 2.0 * (km_expected * u.R_sun).to(u.km).value
    # mas entry would give 5 R_sun, must be ignored
    r_mas_km = 5.0 * (1.0 * u.R_sun).to(u.km).value
    theta_rad = r_mas_km / (d_pc * (1.0 * u.pc).to(u.km).value)
    theta_mas = theta_rad * (648_000.0 / np.pi) * 1e3
    main = _simbad_result({"main_id": "Star", "teff": 4000.0, "plx_value": plx})
    diam = _diam_table([
        {"main_id": "Star", "diameter": d_km,     "unit": "km  "},
        {"main_id": "Star", "diameter": theta_mas, "unit": "mas "},
    ])
    out = _with_diameters(main, diam)
    src = SimbadParamSource()
    src._lookup = src._build_lookup(out)
    result = src.get({"main_id": "Star"})
    assert result["rad"] == pytest.approx(km_expected, rel=1e-4)


def test_no_rad_when_no_diameter_rows():
    # If mesDiameter table is empty, rad must not appear in the lookup
    main = _simbad_result({"main_id": "Star", "teff": 4000.0, "plx_value": 100.0})
    out = _with_diameters(main, _diam_table([]))
    src = SimbadParamSource()
    src._lookup = src._build_lookup(out)
    result = src.get({"main_id": "Star"})
    assert "rad" not in result


def test_mas_without_distance_skipped():
    # mas entry with plx=0 → no distance → rad must be absent
    main = _simbad_result({"main_id": "Star", "teff": 4000.0, "plx_value": 0.0})
    diam = _diam_table([{"main_id": "Star", "diameter": 5.0, "unit": "mas "}])
    out = _with_diameters(main, diam)
    src = SimbadParamSource()
    src._lookup = src._build_lookup(out)
    result = src.get({"main_id": "Star"})
    assert "rad" not in result
