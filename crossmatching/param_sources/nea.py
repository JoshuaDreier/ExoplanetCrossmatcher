import pyvo
from astropy.table import Table

from crossmatching.param_sources.base import StellarParamSource, _build_nea_style_lookup, _safe_err_pair, _safe_float, _safe_str


class NeaStellarParamSource(StellarParamSource):
    """Stellar and planetary params from the NASA Exoplanet Archive (pscomppars).

    Lookup key is the EMC 'nasa_name' column, which maps to 'pl_name' in pscomppars.
    Provides: teff, rad, mass, spec, insol, vmag, dist.
    """

    key_col = "nasa_name"
    source_name = "nea"

    def download(self, key_list: list[str] = None) -> Table:
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        return nasa.run_sync("SELECT * FROM pscomppars").to_table()

    def _build_lookup(self, table: Table) -> dict:
        return _build_nea_style_lookup(table)
