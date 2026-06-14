import pyvo
from astropy.table import Table

from crossmatching.param_sources.base import StellarParamSource, _build_nea_style_lookup, _safe_err_pair, _safe_float, _safe_str


class EpicStellarParamSource(StellarParamSource):
    """Stellar and planetary params from the NASA K2 Candidates and Planets table (k2pandc).

    Covers K2/EPIC planets not present in the main pscomppars table (~68% unique population, 2026-08-06).
    Lookup key is the EMC 'epic_name' column, which maps to 'pl_name' in k2pandc.
    Provides: teff, rad, mass, spec, vmag, dist, insol, logg, met, pl_eqt.
    """

    key_col = "epic_name"
    source_name = "epic"

    def download(self, key_list: list[str] = None) -> Table:
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        return nasa.run_sync("SELECT * FROM k2pandc").to_table()

    def _build_lookup(self, table: Table) -> dict:
        return _build_nea_style_lookup(table)
