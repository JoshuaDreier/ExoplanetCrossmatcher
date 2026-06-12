import astropy.units as u
from astropy.table import Table
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
from tests.functional.conftest import make_catalog

_CATALOG_STAR = {
    "hostname": "Fake Star",
    "pl_name": "Fake Star b",
    "ra": 100.0,
    "dec": 20.0,
}


def _make_cm():
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm._cache_catalog(make_catalog(_CATALOG_STAR))
    return cm


def _input(ra, dec):
    return Table({
        "star_name": ["Test Input"],
        "ra": [ra],
        "dec": [dec],
    })


def test_2d_match_found():
    cm = _make_cm()
    result = cm.coordinate_crossmatch(_input(100.0, 20.0), "star_name")
    assert len(result) > 0
    assert "Fake Star b" in result["pl_name"].tolist()


def test_2d_no_match_when_far():
    cm = _make_cm()
    result = cm.coordinate_crossmatch(_input(110.0, 30.0), "star_name")
    assert len(result) == 0


def test_unknown_search_radius_is_configurable():
    """The synthetic catalog has masked pm/epoch, so every row uses the
    unknown-default radius; a 30 arcsec offset matches at the default 50
    arcsec but not with a 1 arcsec override."""
    offset_30as = 30.0 / 3600.0

    result = _make_cm().coordinate_crossmatch(_input(100.0, 20.0 + offset_30as), "star_name")
    assert len(result) == 1

    cm_tight = Crossmatcher(NEACatalog(), SimbadIdSupplier(), unknown_search_radius=1 * u.arcsec)
    cm_tight._cache_catalog(make_catalog(_CATALOG_STAR))
    result = cm_tight.coordinate_crossmatch(_input(100.0, 20.0 + offset_30as), "star_name")
    assert len(result) == 0


def test_multiple_matches_no_index_mixup():
    """Three input stars each match a different catalog star.
    Verifies each input is paired with its own planet."""
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm._cache_catalog(make_catalog(
        {"hostname": "Alpha", "pl_name": "Alpha b", "ra":  10.0, "dec": 10.0},
        {"hostname": "Beta",  "pl_name": "Beta b",  "ra": 100.0, "dec": 20.0},
        {"hostname": "Gamma", "pl_name": "Gamma b", "ra": 200.0, "dec": 40.0},
    ))
    input_table = Table({
        "star_name": ["in-alpha", "in-beta", "in-gamma"],
        "ra":        [ 10.0,      100.0,      200.0],
        "dec":       [ 10.0,       20.0,       40.0],
    })
    result = cm.coordinate_crossmatch(input_table, "star_name")

    matched = {row["star_name"]: row["pl_name"] for row in result}
    assert matched["in-alpha"] == "Alpha b"
    assert matched["in-beta"]  == "Beta b"
    assert matched["in-gamma"] == "Gamma b"
