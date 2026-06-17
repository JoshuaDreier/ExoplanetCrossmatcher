import pyvo
from astropy.table import Table

from crossmatching.enrichment.param_sources.base import (
    ParamSource,
    _build_nea_style_lookup,
    _safe_err_pair,
    _safe_float,
)


class ToiParamSource(ParamSource):
    """Stellar and planetary params from the NASA TESS Objects of Interest table (toi).

    Covers TESS-discovered planets and candidates not yet in pscomppars (~7900 entries).
    Lookup key is the EMC 'toi_name' column (e.g. 'TOI-651.01'), which matches
    the 'toidisplay' column in the TOI table.
    Provides: teff, rad, dist, insol, logg, pl_eqt.
    Note: TOI uses st_dist (not sy_dist) and has no spectral type or V-band magnitude.
    """

    key_col = "toi_name"
    source_name = "toi"

    param_columns = {
        'teff': 'st_teff',
        'rad': 'st_rad',
        'insol': 'pl_insol',
        'dist': 'st_dist',
        'logg': 'st_logg',
        'pl_eqt': 'pl_eqt',
    }

    param_error_columns = {
        'teff': ('st_tefferr1', 'st_tefferr2'),
        'rad': ('st_raderr1', 'st_raderr2'),
        'insol': ('pl_insolerr1', 'pl_insolerr2'),
        'dist': ('st_disterr1', 'st_disterr2'),
        'logg': ('st_loggerr1', 'st_loggerr2'),
        'pl_eqt': ('pl_eqterr1', 'pl_eqterr2'),
    }

    def download(self, key_list: list[str] = None) -> Table:
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        return nasa.run_sync("SELECT * FROM toi").to_table()

    def _build_lookup(self, table: Table) -> dict:
        return _build_nea_style_lookup(table, self.param_columns, self.param_error_columns, key_col='toidisplay')
