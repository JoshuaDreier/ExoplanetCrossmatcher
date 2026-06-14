import pyvo
from astropy.table import Table

from crossmatching.param_sources.base import StellarParamSource, _safe_err_pair, _safe_float


class ToiStellarParamSource(StellarParamSource):
    """Stellar and planetary params from the NASA TESS Objects of Interest table (toi).

    Covers TESS-discovered planets and candidates not yet in pscomppars (~7900 entries).
    Lookup key is the EMC 'toi_name' column (e.g. 'TOI-651.01'), which matches
    the 'toidisplay' column in the TOI table.
    Provides: teff, rad, dist, insol, logg, pl_eqt.
    Note: TOI uses st_dist (not sy_dist) and has no spectral type or V-band magnitude.
    """

    key_col = "toi_name"
    source_name = "toi"

    def download(self, key_list: list[str] = None) -> Table:
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        return nasa.run_sync("SELECT * FROM toi").to_table()

    def _build_lookup(self, table: Table) -> dict:
        lookup = {}
        cols = set(table.colnames)
        for row in table:
            key = str(row['toidisplay']).strip()
            if key in lookup:
                continue
            entry = {}
            for field, col, positive, e1col, e2col in [
                ('teff',  'st_teff',  True, 'st_tefferr1', 'st_tefferr2'),
                ('rad',   'st_rad',   True, 'st_raderr1',  'st_raderr2'),
                ('insol', 'pl_insol', True, 'pl_insolerr1','pl_insolerr2'),
                # TOI uses st_dist/st_disterr (not sy_dist/sy_disterr)
                ('dist',  'st_dist',  True, 'st_disterr1', 'st_disterr2'),
                ('logg',  'st_logg',  True, 'st_loggerr1', 'st_loggerr2'),
            ]:
                if col not in cols:
                    continue
                v = _safe_float(row[col], require_positive=positive)
                if v is not None:
                    entry[field] = v
                    pair = _safe_err_pair(
                        row[e1col] if e1col in cols else None,
                        row[e2col] if e2col in cols else None,
                    )
                    if pair is not None:
                        entry[f'{field}_err1'], entry[f'{field}_err2'] = pair
            eqt = _safe_float(row['pl_eqt'], require_positive=True)
            if eqt is not None:
                entry['pl_eqt'] = eqt
                pair = _safe_err_pair(
                    row['pl_eqterr1'] if 'pl_eqterr1' in cols else None,
                    row['pl_eqterr2'] if 'pl_eqterr2' in cols else None,
                )
                if pair is not None:
                    entry['pl_eqt_err1'], entry['pl_eqt_err2'] = pair
            lookup[key] = entry
        return lookup
