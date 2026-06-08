from astropy.table import Table

from crossmatching.param_sources.base import StellarParamSource, _safe_float, _safe_str


class HpicStellarParamSource(StellarParamSource):
    """Stellar params from the HPIC crossmatch output (highest-priority tier).

    Constructed with the in-memory crossmatch result table; no download or
    file path is needed since the data is already available after crossmatching.
    """

    key_col = "exo-mercat_name"
    source_name = "hpic"

    def __init__(self, crossmatch_table: Table):
        self._table = crossmatch_table

    def download(self, key_list: list[str]) -> Table:
        raise NotImplementedError("HpicStellarParamSource uses an in-memory table only")

    def load(self, key_list=None, from_file=None, format="ascii") -> Table:
        table = self.preprocess(self._table)
        self._lookup = self._build_lookup(table)
        return table

    def _build_lookup(self, table: Table) -> dict:
        lookup = {}
        for row in table:
            key = str(row[self.key_col]).strip()
            if key in lookup:
                continue
            entry = {}
            teff = _safe_float(row['st_teff'], require_positive=True)
            rad  = _safe_float(row['st_rad'],  require_positive=True)
            spec = _safe_str(row['st_spectype'])
            vmag = _safe_float(row['sy_vmag'])
            if teff is not None: entry['teff'] = teff
            if rad  is not None: entry['rad']  = rad
            if spec is not None: entry['spec'] = spec
            if vmag is not None: entry['vmag'] = vmag
            lookup[key] = entry
        return lookup
