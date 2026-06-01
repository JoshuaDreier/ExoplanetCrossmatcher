from unittest.mock import patch
from astropy.table import Table
import pytest

from crossmatching.id_suppliers.simbad import SimbadIdSupplier


def _mock_response(rows: list[tuple[str, str]]):
    """Build a fake TAP result table with (input_ids, pipe-delimited ids) rows."""
    input_ids, ids = zip(*rows) if rows else ([], [])
    return Table({"input_ids": list(input_ids), "ids": list(ids)})


def _raw_file(tmp_path, rows: list[tuple[str, str]]):
    """Write a raw-format alternate IDs file (input_ids, pipe-delimited ids)."""
    path = tmp_path / "alt_ids.txt"
    _mock_response(rows).write(str(path), format="ascii")
    return str(path)


@patch("crossmatching.id_suppliers.simbad.pyvo.dal.TAPService")
def test_download_splits_pipe_delimited_ids(MockTAP):
    MockTAP.return_value.run_sync.return_value.to_table.return_value = (
        _mock_response([("Star A", "GJ 1|HIP 1|Ross 128")])
    )
    result = SimbadIdSupplier().load_alternate_ids(["Star A"])
    ids = result["id"].tolist()
    assert "GJ 1" in ids
    assert "HIP 1" in ids
    assert "Ross 128" in ids


@patch("crossmatching.id_suppliers.simbad.pyvo.dal.TAPService")
def test_download_preserves_input_id_mapping(MockTAP):
    MockTAP.return_value.run_sync.return_value.to_table.return_value = (
        _mock_response([("Star A", "GJ 1"), ("Star B", "GJ 2")])
    )
    result = SimbadIdSupplier().load_alternate_ids(["Star A", "Star B"])
    a_ids = result["id"][result["input_ids"] == "Star A"].tolist()
    b_ids = result["id"][result["input_ids"] == "Star B"].tolist()
    assert "GJ 1" in a_ids
    assert "GJ 2" in b_ids
    assert "GJ 2" not in a_ids


@patch("crossmatching.id_suppliers.simbad.pyvo.dal.TAPService")
def test_download_expands_star_prefix_variant(MockTAP):
    MockTAP.return_value.run_sync.return_value.to_table.return_value = (
        _mock_response([("Star A", "* Alpha Cen")])
    )
    result = SimbadIdSupplier().load_alternate_ids(["Star A"])
    ids = result["id"].tolist()
    assert "* Alpha Cen" in ids
    assert "Alpha Cen" in ids


def test_file_returns_required_columns(tmp_path):
    path = _raw_file(tmp_path, [("Star A", "GJ 1|HIP 1")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A"], from_file=path)
    assert "input_ids" in result.colnames
    assert "id" in result.colnames


def test_file_splits_pipe_delimited_ids(tmp_path):
    path = _raw_file(tmp_path, [("Star A", "GJ 1|HIP 1|Ross 128")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A"], from_file=path)
    ids = result["id"].tolist()
    assert "GJ 1" in ids
    assert "HIP 1" in ids
    assert "Ross 128" in ids


def test_file_filters_to_name_list(tmp_path):
    path = _raw_file(tmp_path, [("Star A", "GJ 1"), ("Star B", "GJ 2")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A"], from_file=path)
    assert set(result["input_ids"].tolist()) == {"Star A"}
    assert "GJ 2" not in result["id"].tolist()


def test_file_preserves_multiple_stars(tmp_path):
    path = _raw_file(tmp_path, [("Star A", "GJ 1"), ("Star B", "GJ 2")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A", "Star B"], from_file=path)
    assert set(result["input_ids"].tolist()) == {"Star A", "Star B"}


def test_file_expands_star_prefix_variant(tmp_path):
    path = _raw_file(tmp_path, [("Star A", "* Alpha Cen")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A"], from_file=path)
    ids = result["id"].tolist()
    assert "* Alpha Cen" in ids
    assert "Alpha Cen" in ids


def test_file_expands_name_prefix(tmp_path):
    path = _raw_file(tmp_path, [("Star A", "NAME Proxima Cen")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A"], from_file=path)
    ids = result["id"].tolist()
    assert "NAME Proxima Cen" in ids
    assert "Proxima Cen" in ids


def test_file_excludes_empty_ids(tmp_path):
    # Trailing pipe produces an empty token after split; should be dropped
    path = _raw_file(tmp_path, [("Star A", "GJ 1|")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A"], from_file=path)
    assert "" not in result["id"].tolist()


def test_file_excludes_masked_ids(tmp_path):
    # "--" is astropy's masked-value representation for unresolved SIMBAD entries
    path = _raw_file(tmp_path, [("Star A", "--")])
    result = SimbadIdSupplier().load_alternate_ids(["Star A"], from_file=path)
    assert "--" not in result["id"].tolist()


# --- live network ---

@pytest.mark.network
def test_simbad_live_ping():
    # Ross 128 is a stable, well-catalogued nearby star — GJ 447 and HIP 57548 are reliable alternate IDs
    result = SimbadIdSupplier().load_alternate_ids(["Ross 128"])
    ids = result["id"].tolist()
    assert "GJ 447" in ids
    assert "HIP 57548" in ids
