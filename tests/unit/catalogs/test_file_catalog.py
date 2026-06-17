import pytest
import astropy.units as u
from astropy.table import Table

from crossmatching.catalogs.file import FileCatalog


KEYS = dict(
    ra_key="ra",
    ra_unit=u.degree,
    dec_key="dec",
    dec_unit=u.degree,
    host_key="hostname", 
    planet_uid="pl_name"
)


@pytest.fixture
def simple_csv(tmp_path):
    path = tmp_path / "stars.csv"
    Table({
        "hostname":  ["Star A", "Star B"],
        "pl_name":   ["Star A b", "Star B b"],
        "ra":        [10.0, 20.0],
        "dec":       [5.0, 15.0],
    }).write(str(path), format="ascii.csv")
    return str(path)


def test_file_catalog_load_returns_table(simple_csv):
    catalog = FileCatalog(simple_csv, format="ascii.csv", **KEYS)
    result = catalog.load()
    assert isinstance(result, Table)
    assert len(result) == 2


def test_file_catalog_load_has_correct_rows(simple_csv):
    catalog = FileCatalog(simple_csv, format="ascii.csv", **KEYS)
    result = catalog.load()
    assert "Star A b" in result["pl_name"].tolist()
    assert "Star B b" in result["pl_name"].tolist()


def test_file_catalog_no_preprocessing(simple_csv):
    catalog = FileCatalog(simple_csv, format="ascii.csv", **KEYS)
    result = catalog.load()
    assert "coord_epoch" not in result.colnames


def test_file_catalog_load_accepts_override_path(simple_csv, tmp_path):
    other = tmp_path / "other.csv"
    Table({"hostname": ["Star C"], "pl_name": ["Star C b"], "ra": [30.0], "dec": [0.0]}).write(
        str(other), format="ascii.csv"
    )
    catalog = FileCatalog(simple_csv, format="ascii.csv", **KEYS)
    result = catalog.load(from_file=str(other))
    assert result["hostname"].tolist() == ["Star C"]


def test_file_catalog_download_raises():
    with pytest.raises(NotImplementedError):
        FileCatalog("irrelevant.txt", **KEYS).download()


def test_file_catalog_column_keys_stored():
    catalog = FileCatalog("irrelevant.txt", **KEYS)
    assert catalog.ra_key == "ra"
    assert catalog.dec_key == "dec"
    assert catalog.host_key == "hostname"
    assert catalog.planet_uid == "pl_name"
    assert catalog.pm_key is None


def test_file_catalog_accepts_custom_keys(simple_csv):
    catalog = FileCatalog(
        simple_csv,
        format="ascii.csv",
        ra_key="ra",
        ra_unit=u.degree,
        dec_key="dec",
        dec_unit=u.degree,
        host_key="hostname",
        planet_uid="pl_name",
    )
    assert catalog.host_key == "hostname"
    assert catalog.planet_uid == "pl_name"
