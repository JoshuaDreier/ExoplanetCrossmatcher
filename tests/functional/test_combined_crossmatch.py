from astropy.table import Table, Column
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
from tests.functional.conftest import make_catalog


def _make_cm_id_only():
    """Catalog star reachable by ID. Coordinates are masked."""
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name")
    cm._cache_catalog(make_catalog({"hostname": "ID Star", "pl_name": "ID Star b"}))
    cm._cache_alternate_ids(
        Table({"input_ids": ["id-only-input"], "id": ["ID Star"]}),
        ["id-only-input"],
    )
    return cm


def _make_cm_coord_only():
    """Catalog star with no alternate ID and just coordinates."""
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name")
    cm._cache_catalog(make_catalog({"hostname": "Coord Star", "pl_name": "Coord Star b", "ra": 100.0, "dec": 20.0}))
    cm._cache_alternate_ids(
        Table({"input_ids": Column([], dtype="U64"), "id": Column([], dtype="U64")}),
        [],
    )
    return cm


def test_combined_finds_id_match():
    cm = _make_cm_id_only()
    input_table = Table({
        "star_name": ["id-only-input"],
        "ra":        [0.0],
        "dec":       [0.0],
    })
    result = cm.combined_crossmatch(input_table)
    planets = result["pl_name"].tolist()
    match_types = result["match_type"].tolist()
    assert "ID Star b" in planets
    assert match_types[planets.index("ID Star b")] == "id"


def test_combined_finds_coord_match():
    cm = _make_cm_coord_only()
    input_table = Table({
        "star_name": ["coord-only-input"],
        "ra":        [100.0],
        "dec":       [ 20.0],
    })
    result = cm.combined_crossmatch(input_table)
    planets = result["pl_name"].tolist()
    assert "Coord Star b" in planets
    match_types = result["match_type"].tolist()
    assert match_types[planets.index("Coord Star b")] == "coordinates"


def test_combined_deduplicates():
    """A star matched by both ID and coordinates should appear exactly once."""
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name")
    cm._cache_catalog(make_catalog({"hostname": "Both Star", "pl_name": "Both Star b", "ra": 100.0, "dec": 20.0}))
    cm._cache_alternate_ids(
        Table({"input_ids": ["both-match-input"], "id": ["Both Star"]}),
        ["both-match-input"],
    )
    input_table = Table({
        "star_name": ["both-match-input"],
        "ra":        [100.0],
        "dec":       [ 20.0],
    })
    result = cm.combined_crossmatch(input_table)
    assert result["pl_name"].tolist().count("Both Star b") == 1


def test_combined_id_match_favored_over_coord_match():
    """If a star is matched by both ID and coordinates, the match_type should be 'id'."""
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name")
    cm._cache_catalog(make_catalog({"hostname": "Both Star", "pl_name": "Both Star b", "ra": 100.0, "dec": 20.0}))
    cm._cache_alternate_ids(
        Table({"input_ids": ["both-match-input"], "id": ["Both Star"]}),
        ["both-match-input"],
    )
    input_table = Table({
        "star_name": ["both-match-input"],
        "ra":        [100.0],
        "dec":       [ 20.0],
    })
    result = cm.combined_crossmatch(input_table)
    assert result["match_type"][result["star_name"] == "both-match-input"] == ["id"]


def test_combined_one_id_one_2d():
    """Two input stars each matched by a different method.
    id-star:  ID match only (input coords are far from catalog).
    2d-star:  2D match only (same sky position, no alternate ID)."""
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name")
    cm._cache_catalog(make_catalog(
        {"hostname": "ID Star", "pl_name": "ID Star b", "ra": 10.0, "dec": 10.0},
        {"hostname": "2D Star", "pl_name": "2D Star b", "ra": 200.0, "dec": 40.0},
    ))
    cm._cache_alternate_ids(
        Table({"input_ids": ["id-star"], "id": ["ID Star"]}),
        ["id-star", "2d-star"],
    )
    input_table = Table({
        "star_name": ["id-star", "2d-star"],
        "ra":        [  0.0,      200.0],
        "dec":       [  0.0,       40.0],
    })
    result = cm.combined_crossmatch(input_table)

    matched = {row["star_name"]: row["pl_name"] for row in result}
    assert matched["id-star"] == "ID Star b"
    assert matched["2d-star"] == "2D Star b"

    match_type = {row["star_name"]: row["match_type"] for row in result}
    assert match_type["id-star"] == "id"
    assert match_type["2d-star"] == "coordinates"
