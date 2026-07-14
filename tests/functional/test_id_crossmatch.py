from astropy.table import Table, Column
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
from tests.functional.conftest import make_catalog


def _make_cm(catalog_rows, alt_id_pairs, query_names=None):
    """
    Build a Crossmatcher with injected catalog and alternate_ids.
    alt_id_pairs: list of (input_id, catalog_id) tuples.
    query_names: the full set of names that were "queried" (may be a superset of
                 the alt_id_pairs input IDs, for stars that returned no results).
                 Defaults to the input IDs in alt_id_pairs.
    """
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm._cache_catalog(make_catalog(*catalog_rows))
    if alt_id_pairs:
        ids_table = Table({
            "input_ids": [p[0] for p in alt_id_pairs],
            "id":        [p[1] for p in alt_id_pairs],
        })
    else:
        # Explicit string dtype prevents pandas from inferring float64 on empty columns,
        # which would break id_crossmatch's alt_df["id"].str.split() call.
        ids_table = Table({
            "input_ids": Column([], dtype="U64"),
            "id":        Column([], dtype="U64"),
        })
    effective_names = query_names if query_names is not None else [p[0] for p in alt_id_pairs]
    cm._cache_alternate_ids(ids_table, effective_names)
    return cm


def test_id_match_found():
    cm = _make_cm(
        catalog_rows=[{
            "hostname": "Fake Star",
            "pl_name": "Fake Star b",
        }],
        alt_id_pairs=[("fake id", "Fake Star")],
    )
    input_table = Table({"star_name": ["fake id"]})
    result = cm.id_crossmatch(input_table, "star_name")
    assert "Fake Star b" in result["pl_name"].tolist()


def test_id_no_match_when_alternate_ids_empty():
    cm = _make_cm(
        catalog_rows=[{
            "hostname": "Fake Star",
            "pl_name": "Fake Star b",
        }],
        alt_id_pairs=[],
        query_names=["fake id"],
    )
    input_table = Table({"star_name": ["fake id"]})
    result = cm.id_crossmatch(input_table, "star_name")
    assert len(result) == 0


def test_id_multiple_matches_no_index_mixup():
    """Three input stars each map to a different catalog star with a different planet.
    Verifies that each input ID is matched to its own planet, not a neighbour's."""
    cm = _make_cm(
        catalog_rows=[
            {"hostname": "Alpha", "pl_name": "Alpha b"},
            {"hostname": "Beta",  "pl_name": "Beta b"},
            {"hostname": "Gamma", "pl_name": "Gamma b"},
        ],
        alt_id_pairs=[
            ("id-alpha", "Alpha"),
            ("id-beta",  "Beta"),
            ("id-gamma", "Gamma"),
        ],
    )
    input_table = Table({"star_name": ["id-alpha", "id-beta", "id-gamma"]})
    result = cm.id_crossmatch(input_table, "star_name")
    matched = {row["star_name"]: row["pl_name"] for row in result}
    assert matched["id-alpha"] == "Alpha b"
    assert matched["id-beta"]  == "Beta b"
    assert matched["id-gamma"] == "Gamma b"


def test_id_duplicate_alt_id_pairs_do_not_duplicate_planets():
    """Variant expansion can emit the same (input, id) pair from different raw
    IDs (e.g. 'NAME GJ 876' → 'GJ 876' next to a literal 'GJ 876' entry); the
    join must not duplicate matched planet rows."""
    cm = _make_cm(
        catalog_rows=[
            {"hostname": "GJ 876", "pl_name": "GJ 876 b"},
            {"hostname": "GJ 876", "pl_name": "GJ 876 c"},
        ],
        alt_id_pairs=[("fake id", "GJ 876"), ("fake id", "GJ 876")],
    )
    input_table = Table({"star_name": ["fake id"]})
    result = cm.id_crossmatch(input_table, "star_name")
    assert sorted(result["pl_name"].tolist()) == ["GJ 876 b", "GJ 876 c"]


def test_id_no_false_positives():
    cm = _make_cm(
        catalog_rows=[{
            "hostname": "Fake Star A",
            "pl_name": "Fake Star A b",
        },{
            "hostname": "Fake Star B",
            "pl_name": "Fake Star B b",
        },],
        alt_id_pairs=[("fake id 1", "Fake Star A")],
        query_names=["fake id 1", "fake id 2"],
    )
    input_table = Table({"star_name": ["fake id 1", "fake id 2"]})
    result = cm.id_crossmatch(input_table, "star_name")
    planets = result["pl_name"].tolist()
    assert "Fake Star A b" in planets
    assert "Fake Star B b" not in planets
 