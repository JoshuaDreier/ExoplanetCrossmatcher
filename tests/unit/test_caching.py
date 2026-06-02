import pytest
from astropy.table import Table
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier


def _catalog(hostname, pl_name):
    return Table({"hostname": [hostname], "pl_name": [pl_name]})


@pytest.fixture
def cm():
    instance = Crossmatcher(NEACatalog(), SimbadIdSupplier(), input_starname_key="star_name")
    instance._cache_catalog(_catalog("GJ 1", "GJ 1 b"))
    return instance


def test_id_crossmatch_reloads_alt_ids_for_disjoint_input(cm, monkeypatch):
    """
    Calling id_crossmatch twice with disjoint star sets must reload alt IDs
    for the second call, not silently reuse the first call's data (Bug A).
    """
    call_count = 0

    def tracking_load(name_list, **kwargs):
        nonlocal call_count
        call_count += 1
        if "Star A" in name_list:
            return Table({"input_ids": ["Star A"], "id": ["GJ 1"]})
        return Table({"input_ids": ["Star B"], "id": ["GJ 999"]})

    monkeypatch.setattr(cm.id_supplier, "load_alternate_ids", tracking_load)

    cm.id_crossmatch(Table({"star_name": ["Star A"]}))
    assert call_count == 1

    result_b = cm.id_crossmatch(Table({"star_name": ["Star B"]}))
    assert call_count == 2, "alt IDs were not reloaded for the new input (Bug A)"
    assert "Star A" not in result_b["star_name"].tolist()


def test_id_crossmatch_does_not_reload_for_same_input(cm, monkeypatch):
    """Cache should be reused when the same name list is passed again."""
    call_count = 0

    def tracking_load(name_list, **kwargs):
        nonlocal call_count
        call_count += 1
        return Table({"input_ids": ["Star A"], "id": ["GJ 1"]})

    monkeypatch.setattr(cm.id_supplier, "load_alternate_ids", tracking_load)

    input_a = Table({"star_name": ["Star A"]})
    cm.id_crossmatch(input_a)
    cm.id_crossmatch(input_a)
    assert call_count == 1, "alt IDs were reloaded unnecessarily for identical input"


def test_id_crossmatch_trims_cache_for_subset(cm, monkeypatch):
    """A subset input should trim the cached data, not reload from source."""
    call_count = 0

    def tracking_load(name_list, **kwargs):
        nonlocal call_count
        call_count += 1
        return Table({
            "input_ids": ["Star A", "Star B"],
            "id": ["GJ 1", "GJ 2"],
        })

    monkeypatch.setattr(cm.id_supplier, "load_alternate_ids", tracking_load)

    cm.id_crossmatch(Table({"star_name": ["Star A", "Star B"]}))
    assert call_count == 1

    cm.id_crossmatch(Table({"star_name": ["Star A"]}))
    assert call_count == 1, "cache was reloaded for a strict subset (should trim)"
    assert set(cm.alternate_ids["input_ids"].tolist()) == {"Star A"}
