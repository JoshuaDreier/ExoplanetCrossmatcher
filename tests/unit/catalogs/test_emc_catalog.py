import os
import pytest
import numpy as np
from astropy.table import Table, Column
from crossmatching import Crossmatcher, EMCCatalog, SimbadIdSupplier


_EMC_FILE = "exo-mercat.csv"
_REQUIRED_COLUMNS = ("host", "main_id_ra", "main_id_dec", "exo-mercat_name", "main_id_aliases")


@pytest.mark.skipif(not os.path.exists(_EMC_FILE), reason=f"{_EMC_FILE} not present")
def test_load_from_file():
    cat = EMCCatalog().load(from_file=_EMC_FILE, format="csv")
    assert len(cat) > 0
    for col in _REQUIRED_COLUMNS:
        assert col in cat.colnames, f"Missing column: {col}"


def test_no_pm_fallback_in_coordinate_crossmatch():
    # if EMCCatalog has pm_key=None, then coordinate_crossmatch must fall back to
    # unknown_default (50 arcsec) for all rows without raising.
    cat = Table({
        "host": Column(["Test Host"]),
        "exo-mercat_name": Column(["Test Host b"]),
        "main_id_ra": Column([180.0]),
        "main_id_dec":Column([0.0]),
    })
    inp = Table({
        "star_name": ["TIC 1"],
        "ra":        [180.0],
        "dec":       [0.0],
    })

    cm = Crossmatcher(EMCCatalog(), SimbadIdSupplier())
    cm._cache_catalog(cat)
    result = cm.coordinate_crossmatch(inp, input_starname_key="star_name")

    # The 0-separation match must fire (separation 0 ≤ unknown_default 50 arcsec).
    assert len(result) == 1
    assert result["exo-mercat_name"][0] == "Test Host b"
    assert float(result[cm.angular_sep_key][0]) == pytest.approx(0.0, abs=1e-5)
