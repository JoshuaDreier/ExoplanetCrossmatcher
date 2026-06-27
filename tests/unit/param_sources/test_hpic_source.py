import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from crossmatching.enrichment.param_sources.hpic import HpicParamSource


def _crossmatch_table(*rows):
    """Minimal crossmatch-output table accepted by HpicParamSource."""
    names = [r["name"] for r in rows]
    teff  = [r.get("st_teff",   0.0) for r in rows]
    rad   = [r.get("st_rad",    0.0) for r in rows]
    spec  = [r.get("st_spectype", "") for r in rows]
    vmag  = [r.get("sy_vmag",  np.nan) for r in rows]
    return Table({
        "exo-mercat_name": names,
        "st_teff":          teff,
        "st_rad":           rad,
        "st_spectype":      spec,
        "sy_vmag":          vmag,
    })


def _loaded_source(*rows):
    src = HpicParamSource(_crossmatch_table(*rows))
    src.load()
    return src


def _make_row(name):
    """Minimal catalog row for get()."""
    return {"exo-mercat_name": name}


def test_returns_known_params():
    src = _loaded_source({"name": "Star b", "st_teff": 5500.0, "st_rad": 0.9, "sy_vmag": 8.2})
    result = src.get(_make_row("Star b"))
    assert result["teff"] == pytest.approx(5500.0)
    assert result["rad"]  == pytest.approx(0.9)
    assert result["vmag"] == pytest.approx(8.2)


def test_returns_empty_for_unknown_name():
    src = _loaded_source({"name": "Star b", "st_teff": 5500.0, "st_rad": 0.9})
    assert src.get(_make_row("Unknown c")) == {}


def test_zero_teff_not_included():
    # teff=0 is a sentinel "no data"; should not appear in the returned dict
    src = _loaded_source({"name": "Star b", "st_teff": 0.0, "st_rad": 0.9})
    result = src.get(_make_row("Star b"))
    assert "teff" not in result


def test_zero_rad_not_included():
    src = _loaded_source({"name": "Star b", "st_teff": 5500.0, "st_rad": 0.0})
    result = src.get(_make_row("Star b"))
    assert "rad" not in result


def test_first_occurrence_wins_on_duplicate_name():
    # Two rows with the same exo-mercat_name — first should be used
    src = HpicParamSource(_crossmatch_table(
        {"name": "Dup b", "st_teff": 4000.0, "st_rad": 0.7},
        {"name": "Dup b", "st_teff": 9999.0, "st_rad": 9.9},
    ))
    src.load()
    assert src.get(_make_row("Dup b"))["teff"] == pytest.approx(4000.0)


def test_download_raises():
    src = HpicParamSource(_crossmatch_table({"name": "X b"}))
    with pytest.raises(NotImplementedError):
        src.download([])
