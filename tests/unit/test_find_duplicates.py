import pytest
from tests.unit.unit_fixtures import cm_with_duplicates, input_table


def test_find_duplicates_not_full(input_table, cm_with_duplicates):
    result = cm_with_duplicates.find_duplicates(input_table, full=False)
    assert len(result) == 1
    assert "Star A" in result[0]["duplicate_names"]
    assert "Star B" in result[0]["duplicate_names"]

def test_find_duplicates_full(input_table, cm_with_duplicates):
    result = cm_with_duplicates.find_duplicates(input_table, full=True)
    assert len(result) == 2
    for row in result:
        assert "Star A" in row["duplicate_names"]
        assert "Star B" in row["duplicate_names"]

def test_find_duplicates_no_false_positives(cm_with_duplicates, input_table):
    result = cm_with_duplicates.find_duplicates(input_table)
    for row in result:
        assert "Star C" not in row["duplicate_names"]
