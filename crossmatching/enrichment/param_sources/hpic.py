from astropy.table import Table

from crossmatching.enrichment.param_sources.base import ParamSource, _build_nea_style_lookup, _safe_float, _safe_str
import numpy as np


class HpicParamSource(ParamSource):
    """Stellar params from the HPIC crossmatch output (highest-priority tier).

    Constructed with the in-memory crossmatch result table; no download or
    file path is needed since the data is already available after crossmatching.
    """

    key_col = "exo-mercat_name"
    source_name = "hpic"

    param_columns = {
        'teff': 'st_teff',
        'rad': 'st_rad',
        'spec': 'st_spectype',
        'vmag': 'sy_vmag',
    }

    def __init__(self, crossmatch_table: Table):
        self._table = crossmatch_table

    def download(self, key_list: list[str]) -> Table:
        raise NotImplementedError("HpicStellarParamSource uses an in-memory table only")

    def load(self, key_list=None, from_file=None, format="ascii") -> Table:
        table = self.preprocess(self._table)
        self._lookup = self._build_lookup(table)
        return table

    param_error_columns = {}

    def _build_lookup(self, table: Table) -> dict:
        """Build lookup using unified NEA‑style helper.

        ``exo‑mercat_name`` is the identifier column, so we pass it via
        ``key_col``.  No error columns are defined for this source.
        """
        return _build_nea_style_lookup(table, self.param_columns, self.param_error_columns, key_col=self.key_col)
