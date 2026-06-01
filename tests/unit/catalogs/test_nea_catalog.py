import numpy as np
import numpy.ma as ma
from astropy.table import Table, MaskedColumn

from crossmatching.catalogs.nea import NEACatalog, _coord_epoch



class TestNEACatalogKeys:
    def test_ra_key(self):
        assert NEACatalog().ra_key == "ra"

    def test_dec_key(self):
        assert NEACatalog().dec_key == "dec"

    def test_hostname_key(self):
        assert NEACatalog().host_key == "hostname"

    def test_planet_uuid(self):
        assert NEACatalog().planet_uuid == "pl_name"

    def test_pm_key(self):
        assert NEACatalog().pm_key == "sy_pm"

    def test_pmerr_key(self):
        assert NEACatalog().pmerr_key == "sy_pmerr1"


class TestCoordEpoch:
    def test_gaia_dr3_returns_2016(self):
        assert _coord_epoch("", "1234567890", "") == 2016.0

    def test_gaia_dr2_returns_2016(self):
        assert _coord_epoch("", "", "987654321") == 2016.0

    def test_stassun_reflink_returns_2000(self):
        assert _coord_epoch("refstr=STASSUN_ET_AL__2019", "", "") == 2000.0

    def test_ticv8_reflink_returns_2000(self):
        assert _coord_epoch("TICv8 survey", "", "") == 2000.0

    def test_hipparcos_reflink_returns_1991(self):
        assert _coord_epoch("Hipparcos catalog", "", "") == 1991.25

    def test_post_2018_pub_year_returns_2016(self):
        assert _coord_epoch("refstr=AUTHOR_2020", "", "") == 2016.0

    def test_pre_2018_pub_year_returns_that_year(self):
        assert _coord_epoch("refstr=AUTHOR_2015", "", "") == 2015.0

    def test_unknown_returns_none(self):
        assert _coord_epoch("", "", "") is None

    def test_masked_reflink_returns_none(self):
        assert _coord_epoch(np.ma.masked, np.ma.masked, np.ma.masked) is None


class TestNEACatalogLoad:
    def test_load_from_file_has_required_columns(self):
        cat = NEACatalog().load(from_file="pscomppars.txt")
        for col in ("hostname", "pl_name", "ra", "dec", "sy_pm", "sy_pmerr1"):
            assert col in cat.colnames

    def test_load_adds_coord_epoch_column(self):
        cat = NEACatalog().load(from_file="pscomppars.txt")
        assert "coord_epoch" in cat.colnames

    def test_coord_epoch_known_rows_have_plausible_value(self):
        cat = NEACatalog().load(from_file="pscomppars.txt")
        has_epoch = ~np.ma.getmaskarray(cat["coord_epoch"])
        assert float(cat["coord_epoch"][has_epoch][0]) in (1991.25, 2000.0, 2016.0)

    def test_load_raw_does_not_add_coord_epoch(self):
        raw = NEACatalog().load_raw("pscomppars.txt")
        assert "coord_epoch" not in raw.colnames

    def test_preprocess_adds_coord_epoch(self):
        raw = NEACatalog().load_raw("pscomppars.txt")
        processed = NEACatalog().preprocess(raw)
        assert "coord_epoch" in processed.colnames

