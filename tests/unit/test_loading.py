from crossmatching import Crossmatcher
from pytest_check import check


def test_load_catalog_has_required_columns():
    cm = Crossmatcher()
    cm.load_catalog(from_file="pscomppars.txt")
    for column_name in ["pl_name", "hostname", "ra", "dec", "sy_dist"]:
        check.is_in(column_name, cm.catalogue.colnames)


def test_load_catalog_nonempty():
    cm = Crossmatcher()
    cm.load_catalog(from_file="pscomppars.txt")
    assert len(cm.catalogue) > 0


def test_load_alternate_ids_has_required_columns():
    cm = Crossmatcher()
    cm.load_alternate_ids(["TIC 325275315"], from_file="alternate_ids.txt")
    for column_name in ["input_ids", "id"]:
        check.is_in(column_name, cm.alternate_ids.colnames)


def test_load_alternate_ids_filtered_to_name_list():
    cm = Crossmatcher()
    cm.load_alternate_ids(["TIC 325275315"], from_file="alternate_ids.txt")
    assert len(cm.alternate_ids) > 0
    assert all(row["input_ids"] == "TIC 325275315" for row in cm.alternate_ids)


