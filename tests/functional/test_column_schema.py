"""
Tests that id_crossmatch and coordinate_crossmatch produce tables with a
consistent column schema, and that combined_crossmatch inherits that schema.

Neither method should rename shared columns with _input / _cat suffixes —
the output columns should be the same identifiers that appear in the input
table and the catalog (ra, dec, …), not ra_input / ra_cat.
"""

from astropy.table import Table
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
from tests.functional.conftest import make_catalog

_CATALOG = {"hostname": "Schema Star", "pl_name": "Schema Star b", "ra": 100.0, "dec": 20.0}

_INPUT = Table({
    "star_name": ["schema-input"],
    "ra":        [100.0],
    "dec":       [20.0],
})


def _make_cm():
    """Crossmatcher where the single star is reachable by both ID and coordinates."""
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm._cache_catalog(make_catalog(_CATALOG))
    cm._cache_alternate_ids(
        Table({"input_ids": ["schema-input"], "id": ["Schema Star"]}),
        ["schema-input"],
    )
    return cm


def test_id_and_coord_output_columns_match():
    """id_crossmatch and coordinate_crossmatch must produce the same column set.

    The only acceptable differences are method-specific bookkeeping columns:
    angular_separation (coordinate only) and coord_epoch (coordinate only).
    The core columns — star_name, ra, dec, hostname, pl_name, match_type, and
    all other catalog columns — must be identical in both outputs so that
    combined_crossmatch can vstack them without gaps or duplication.
    """
    cm_id    = _make_cm()
    cm_coord = _make_cm()

    id_result    = cm_id.id_crossmatch(_INPUT)
    coord_result = cm_coord.coordinate_crossmatch(_INPUT)

    coord_only_extras = {"angular_separation", "coord_epoch"}

    id_cols    = set(id_result.colnames)    - coord_only_extras
    coord_cols = set(coord_result.colnames) - coord_only_extras

    assert id_cols == coord_cols, (
        "id_crossmatch and coordinate_crossmatch column sets differ after "
        "excluding method-specific columns.\n"
        f"  in id but not coord:    {sorted(id_cols - coord_cols)}\n"
        f"  in coord but not id:    {sorted(coord_cols - id_cols)}"
    )


def test_combined_crossmatch_column_schema():
    """combined_crossmatch output column schema:

    - Catalog columns keep their original names (ra, dec, …).
    - Input columns that share a name with the catalog are present under the
      {col}_input suffix (ra_input, dec_input, …).
    - No _cat suffix appears anywhere — only the input side is ever suffixed.
    - Mandatory bookkeeping and identifier columns are present.
    """
    cm = _make_cm()
    result = cm.combined_crossmatch(_INPUT)
    colnames = result.colnames

    suffixed_cat = [c for c in colnames if c.endswith("_cat")]
    assert not suffixed_cat, (
        f"combined_crossmatch produced _cat-suffixed columns: {suffixed_cat}"
    )

    for expected in ("star_name", "ra", "dec", "ra_input", "dec_input",
                     "hostname", "pl_name", "match_type"):
        assert expected in colnames, (
            f"Expected column '{expected}' missing from combined_crossmatch output. "
            f"Got: {colnames}"
        )
