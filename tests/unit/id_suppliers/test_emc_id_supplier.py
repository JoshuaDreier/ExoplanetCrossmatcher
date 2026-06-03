import pandas as pd
import pytest
from astropy.table import Table

from crossmatching.id_suppliers.emc import EMCIdSupplier


def _raw_table(*rows):
    """Minimal two-column table accepted by EMCIdSupplier.preprocess."""
    hosts, aliases = zip(*rows) if rows else ([], [])
    return Table({"host": list(hosts), "main_id_aliases": list(aliases)})


def _csv_file(tmp_path, *rows):
    """Write a minimal EMC-shaped CSV for load_alternate_ids tests."""
    hosts, aliases = zip(*rows) if rows else ([], [])
    df = pd.DataFrame({"host": list(hosts), "main_id_aliases": list(aliases)})
    path = tmp_path / "emc.csv"
    df.to_csv(str(path), index=False)
    return str(path)


def test_preprocess_basic():
    raw = _raw_table(("Tau Cet", "TIC 111,HD 10700"))
    result = EMCIdSupplier().preprocess(raw)
    ids = result["id"].tolist()
    inputs = result["input_ids"].tolist()
    assert "TIC 111" in inputs
    assert "HD 10700" in inputs
    assert all(h == "Tau Cet" for h in ids)


def test_preprocess_deduplicates_by_host():
    # Three planet rows for the same host, but aliases should be emitted only once.
    single = EMCIdSupplier().preprocess(_raw_table(("Tau Cet", "TIC 111,HD 10700")))
    triple = EMCIdSupplier().preprocess(_raw_table(
        ("Tau Cet", "TIC 111,HD 10700"),
        ("Tau Cet", "TIC 111,HD 10700"),
        ("Tau Cet", "TIC 111,HD 10700"),
    ))
    assert len(triple) == len(single)


def test_star_prefix_stripped():
    # Multi-space is normalized first ("*   Tau Cet" → "* Tau Cet"), then the
    # prefix is stripped to produce the additional variant "Tau Cet".
    raw = _raw_table(("Tau Cet", "*   Tau Cet"))
    result = EMCIdSupplier().preprocess(raw)
    inputs = result["input_ids"].tolist()
    assert "* Tau Cet" in inputs
    assert "Tau Cet" in inputs


def test_name_prefix_stripped():
    raw = _raw_table(("Proxima", "NAME Proxima Cen"))
    result = EMCIdSupplier().preprocess(raw)
    inputs = result["input_ids"].tolist()
    assert "NAME Proxima Cen" in inputs
    assert "Proxima Cen" in inputs


def test_possessive_stripped():
    raw = _raw_table(("Barnard", "Barnard's"))
    result = EMCIdSupplier().preprocess(raw)
    inputs = result["input_ids"].tolist()
    assert "Barnard's" in inputs
    assert "Barnard" in inputs


def test_gaia_case_expanded():
    # SIMBAD stores "Gaia DR3 ..." but HPIC uses "GAIA DR3 ...".
    raw = _raw_table(("Wolf 359", "Gaia DR3 3864972938605115520"))
    result = EMCIdSupplier().preprocess(raw)
    inputs = result["input_ids"].tolist()
    assert "Gaia DR3 3864972938605115520" in inputs
    assert "GAIA DR3 3864972938605115520" in inputs


def test_null_aliases_skipped():
    raw = _raw_table(
        ("Empty host", ""),
        ("Sentinel host", "--"),
        ("Valid host", "HD 1234"),
    )
    result = EMCIdSupplier().preprocess(raw)
    assert "Empty host" not in result["id"].tolist()
    assert "Sentinel host" not in result["id"].tolist()
    assert "Valid host" in result["id"].tolist()


def test_load_alternate_ids_filters_by_name_list(tmp_path):
    path = _csv_file(tmp_path,
        ("Alpha Star", "TIC 111"),
        ("Beta Star",  "TIC 222"),
        ("Gamma Star", "TIC 333"),
    )
    result = EMCIdSupplier().load_alternate_ids(["TIC 111"], from_file=path)
    assert set(result["input_ids"].tolist()) == {"TIC 111"}
    assert "TIC 222" not in result["input_ids"].tolist()
    assert "TIC 333" not in result["input_ids"].tolist()
