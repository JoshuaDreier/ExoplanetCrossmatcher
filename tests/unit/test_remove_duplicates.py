from astropy.table import Table
from tests.unit.unit_fixtures import cm_with_duplicates


def test_remove_duplicates_reduces_row_count(cm_with_duplicates):
    input_table = Table({
        "star_name": ["Star A", "Star B", "Star C"],
        "sy_dist": ["10.0", "10.0", "5.0"],
    })
    result = cm_with_duplicates.remove_duplicates(input_table, "star_name")
    assert len(result) == len(input_table) - 1


def test_remove_duplicates_keeps_more_complete_row(cm_with_duplicates):
    """Star B have an empty sy_dist (one extra null) so Star B should survive."""
    input_table = Table({
        "star_name": ["Star A", "Star B", "Star C"],
        "sy_dist": ["", "1.0", "5.0"],
    })
    result = cm_with_duplicates.remove_duplicates(input_table, "star_name")
    star_names = result["star_name"].tolist()
    assert "Star B" in star_names
    assert "Star C" in star_names
    assert "Star A" not in star_names