"""
Tests for the configurable schema keys on Crossmatcher.
Each test pairs a custom kwarg against the specific behaviour it controls,
verifying the kwarg flows through to the output rather than being ignored.
"""
import pytest
from astropy.table import Table, Column
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
from tests.functional.conftest import make_catalog


_CATALOG_ID = make_catalog({"hostname": "Star X", "pl_name": "Star X b"})
_CATALOG_COORD = make_catalog({"hostname": "Star X", "pl_name": "Star X b", "ra": 100.0, "dec": 20.0})

_ALT_IDS = Table({"input_ids": ["test-star"], "id": ["Star X"]})
_ALT_IDS_EMPTY = Table({"input_ids": Column([], dtype="U64"), "id": Column([], dtype="U64")})

_ID_INPUT = Table({"star_name": ["test-star"]})
_COORD_INPUT = Table({"star_name": ["test-input"], "ra": [100.0], "dec": [20.0]})


def _cm_id(**kwargs):
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name", **kwargs)
    cm._cache_catalog(_CATALOG_ID)
    cm._cache_alternate_ids(_ALT_IDS, ["test-star"])
    return cm


def _cm_coord(**kwargs):
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name", **kwargs)
    cm._cache_catalog(_CATALOG_COORD)
    cm._cache_alternate_ids(_ALT_IDS_EMPTY, [])
    return cm


def test_input_starname_key_is_required():
    with pytest.raises(TypeError):
        Crossmatcher(NEACatalog(), SimbadIdSupplier())


def test_custom_match_type_key_id():
    result = _cm_id(match_type_key="my_type").id_crossmatch(_ID_INPUT)
    assert "my_type" in result.colnames
    assert "match_type" not in result.colnames


def test_custom_match_type_key_coord():
    result = _cm_coord(match_type_key="my_type").coordinate_crossmatch(_COORD_INPUT)
    assert "my_type" in result.colnames
    assert "match_type" not in result.colnames


def test_custom_id_match_label():
    result = _cm_id(id_match_label="identifier").id_crossmatch(_ID_INPUT)
    assert result["match_type"][0] == "identifier"


def test_custom_coord_match_label():
    result = _cm_coord(coord_match_label="sky").coordinate_crossmatch(_COORD_INPUT)
    assert result["match_type"][0] == "sky"


def test_custom_angular_sep_key():
    result = _cm_coord(angular_sep_key="sep_arcsec").coordinate_crossmatch(_COORD_INPUT)
    assert "sep_arcsec" in result.colnames
    assert "angular_separation" not in result.colnames


def test_custom_input_suffix_coord():
    result = _cm_coord(input_suffix="hpic").coordinate_crossmatch(_COORD_INPUT)
    assert "ra_hpic" in result.colnames
    assert "ra_input" not in result.colnames


def test_custom_input_suffix_id():
    input_with_ra = Table({"star_name": ["test-star"], "ra": [0.0]})
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name", input_suffix="hpic")
    cm._cache_catalog(_CATALOG_ID)
    cm._cache_alternate_ids(_ALT_IDS, ["test-star"])
    result = cm.id_crossmatch(input_with_ra)
    assert "ra_hpic" in result.colnames
    assert "ra_input" not in result.colnames
