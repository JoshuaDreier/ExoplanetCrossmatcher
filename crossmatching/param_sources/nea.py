import pyvo
from astropy.table import Table

from crossmatching.param_sources.base import StellarParamSource, _safe_err_pair, _safe_float, _safe_str


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
        lookup = {}
        cols = set(table.colnames)
        for row in table:
            key = str(row['pl_name']).strip()
            if key in lookup:
                continue
            entry = {}
            for field, col, positive, e1col, e2col in [
                ('teff',  'st_teff',  True,  'st_tefferr1',  'st_tefferr2'),
                ('rad',   'st_rad',   True,  'st_raderr1',   'st_raderr2'),
                ('mass',  'st_mass',  True,  'st_masserr1',  'st_masserr2'),
                ('insol', 'pl_insol', True,  'pl_insolerr1', 'pl_insolerr2'),
                ('vmag',  'sy_vmag',  False, 'sy_vmagerr1',  'sy_vmagerr2'),
                ('dist',  'sy_dist',  True,  'sy_disterr1',  'sy_disterr2'),
                ('logg',  'st_logg',  True,  'st_loggerr1',  'st_loggerr2'),
            ]:
                v = _safe_float(row[col], require_positive=positive)
                if v is not None:
                    entry[field] = v
                    pair = _safe_err_pair(
                        row[e1col] if e1col in cols else None,
                        row[e2col] if e2col in cols else None,
                    )
                    if pair is not None:
                        entry[f'{field}_err1'], entry[f'{field}_err2'] = pair
            # metallicity: 0.0 is physically valid (solar), require_positive=False
            met = _safe_float(row['st_met'])
            if met is not None:
                entry['met'] = met
                pair = _safe_err_pair(
                    row['st_meterr1'] if 'st_meterr1' in cols else None,
                    row['st_meterr2'] if 'st_meterr2' in cols else None,
                )
                if pair is not None:
                    entry['met_err1'], entry['met_err2'] = pair
            eqt = _safe_float(row['pl_eqt'], require_positive=True) if 'pl_eqt' in cols else None
            if eqt is not None:
                entry['pl_eqt'] = eqt
                pair = _safe_err_pair(
                    row['pl_eqterr1'] if 'pl_eqterr1' in cols else None,
                    row['pl_eqterr2'] if 'pl_eqterr2' in cols else None,
                )
                if pair is not None:
                    entry['pl_eqt_err1'], entry['pl_eqt_err2'] = pair
            spec = _safe_str(row['st_spectype'])
            if spec is not None:
                entry['spec'] = spec
            lookup[key] = entry
        return lookup
