import pytest
from astropy.table import Table
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
from pytest_check import check


def test_load_catalog_has_required_columns():
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm.load_catalog(from_file="pscomppars.txt")
    for column_name in ["pl_name", "hostname", "ra", "dec", "sy_dist"]:
        check.is_in(column_name, cm.catalog_table.colnames)


def test_load_catalog_nonempty():
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm.load_catalog(from_file="pscomppars.txt")
    assert len(cm.catalog_table) > 0


def test_load_alternate_ids_has_required_columns(tmp_path):
    path = tmp_path / "alt_ids.txt"
    Table({"input_ids": ["TIC 325275315"], "ids": ["HIP 1|GJ 1"]}).write(
        str(path), format="ascii"
    )
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm.load_alternate_ids(["TIC 325275315"], from_file=str(path))
    for column_name in ["input_ids", "id"]:
        check.is_in(column_name, cm.alternate_ids.colnames)


def test_load_alternate_ids_preprocesses_from_file(tmp_path):
    path = tmp_path / "alt_ids.txt"
    Table({"input_ids": ["TIC 325275315"], "ids": ["HIP 1|GJ 1"]}).write(
        str(path), format="ascii"
    )
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm.load_alternate_ids(["TIC 325275315"], from_file=str(path))
    assert len(cm.alternate_ids) > 0
    assert "HIP 1" in cm.alternate_ids["id"].tolist()
    assert "GJ 1" in cm.alternate_ids["id"].tolist()


