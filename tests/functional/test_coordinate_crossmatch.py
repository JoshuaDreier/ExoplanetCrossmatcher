from astropy.table import Table
from crossmatching import Crossmatcher
from tests.functional.conftest import make_catalog

_CATALOG_STAR = {
    "hostname": "Fake Star", 
    "pl_name": "Fake Star b",
    "ra": 100.0,
    "dec": 20.0,
    "sy_dist": 10.0
}


def _make_cm():
    cm = Crossmatcher()
    cm.catalogue = make_catalog(_CATALOG_STAR)
    cm.catalogue_cached = True
    return cm


def _input(ra, dec, sy_dist=10.0):
    return Table({
        "star_name": ["Test Input"], 
        "ra": [ra], 
        "dec": [dec],
        "sy_dist": [sy_dist]
    })


def test_2d_match_found():
    cm = _make_cm()
    result3d, result2d = cm.coordinate_crossmatch(_input(100.0, 20.0))
    assert len(result2d) > 0
    assert "Fake Star b" in result2d["pl_name"].tolist()


def test_2d_no_match_when_far():
    cm = _make_cm()
    _, result2d = cm.coordinate_crossmatch(_input(110.0, 30.0))
    assert len(result2d) == 0


def test_3d_match_found():
    cm = _make_cm()
    result3d, _ = cm.coordinate_crossmatch(_input(100.0, 20.0, sy_dist=10.0))
    assert len(result3d) > 0
    assert "Fake Star b" in result3d["pl_name"].tolist()


def test_3d_no_match_wrong_distance():
    """Same sky position but very different distance — 2D matches but 3D should not."""
    cm = _make_cm()
    result3d, _ = cm.coordinate_crossmatch(_input(100.0, 20.0, sy_dist=20.0))
    assert len(result3d) == 0


def test_multiple_matches_no_index_mixup():
    """Three input stars each match a different catalog star.
    Verifies each input is paired with its own planet in both 2D and 3D results."""
    cm = Crossmatcher()
    cm.catalogue = make_catalog(
        {"hostname": "Alpha", "pl_name": "Alpha b", "ra":  10.0, "dec": 10.0, "sy_dist": 10.0},
        {"hostname": "Beta",  "pl_name": "Beta b",  "ra": 100.0, "dec": 20.0, "sy_dist": 20.0},
        {"hostname": "Gamma", "pl_name": "Gamma b", "ra": 200.0, "dec": 40.0, "sy_dist": 30.0},
    )
    cm.catalogue_cached = True
    input_table = Table({
        "star_name": ["in-alpha", "in-beta", "in-gamma"],
        "ra":        [ 10.0,      100.0,      200.0],
        "dec":       [ 10.0,       20.0,       40.0],
        "sy_dist":   [ 10.0,       20.0,       30.0],
    })
    result3d, result2d = cm.coordinate_crossmatch(input_table)

    matched2d = {row["star_name"]: row["pl_name"] for row in result2d}
    assert matched2d["in-alpha"] == "Alpha b"
    assert matched2d["in-beta"]  == "Beta b"
    assert matched2d["in-gamma"] == "Gamma b"

    matched3d = {row["star_name"]: row["pl_name"] for row in result3d}
    assert matched3d["in-alpha"] == "Alpha b"
    assert matched3d["in-beta"]  == "Beta b"
    assert matched3d["in-gamma"] == "Gamma b"
