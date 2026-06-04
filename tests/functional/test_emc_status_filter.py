from astropy.table import Table, Column
from crossmatching import Crossmatcher, EMCCatalog, SimbadIdSupplier


def _make_emc_cm(allowed_statuses):
    emc_raw = Table({
        "host":            Column(["Alpha", "Alpha", "Alpha"]),
        "exo-mercat_name": Column(["Alpha b", "Alpha c", "Alpha d"]),
        "main_id_ra":      Column([180.0, 180.0, 180.0]),
        "main_id_dec":     Column([0.0, 0.0, 0.0]),
        "status":          Column(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]),
    })
    catalog = EMCCatalog(allowed_statuses=allowed_statuses)
    cm = Crossmatcher(catalog, SimbadIdSupplier())
    cm._cache_catalog(catalog.preprocess(emc_raw))
    cm._cache_alternate_ids(
        Table({"input_ids": ["alpha-input"], "id": ["Alpha"]}),
        ["alpha-input"],
    )
    return cm


_INPUT = Table({"star_name": ["alpha-input"], "ra": [0.0], "dec": [0.0]})


def test_default_allowed_statuses_excludes_false_positives():
    result = _make_emc_cm(None).id_crossmatch(_INPUT, input_starname_key="star_name")
    planets = result["exo-mercat_name"].tolist()
    assert "Alpha b" in planets
    assert "Alpha c" not in planets  # FALSE POSITIVE excluded by default
    assert "Alpha d" in planets      # CANDIDATE kept by default


def test_custom_confirmed_only():
    result = _make_emc_cm(["CONFIRMED"]).id_crossmatch(_INPUT, input_starname_key="star_name")
    planets = result["exo-mercat_name"].tolist()
    assert "Alpha b" in planets
    assert "Alpha c" not in planets
    assert "Alpha d" not in planets


def test_explicit_all_statuses_keeps_everything():
    result = _make_emc_cm(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]).id_crossmatch(_INPUT, input_starname_key="star_name")
    planets = result["exo-mercat_name"].tolist()
    assert "Alpha b" in planets
    assert "Alpha c" in planets
    assert "Alpha d" in planets
