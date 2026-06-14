import numpy as np
from astropy.table import Table


def _build_nea_style_lookup(table: Table) -> dict:
    """Build a pl_name-keyed parameter lookup from an NEA-schema table.

    Shared by NeaStellarParamSource and EpicStellarParamSource, which use
    identical pscomppars-style column layouts.  All column accesses are
    guarded against schema differences via ``if col in cols``.
    """
    lookup: dict = {}
    cols = set(table.colnames)
    for row in table:
        key = str(row['pl_name']).strip()
        if key in lookup:
            continue
        entry: dict = {}
        for field, col, positive, e1col, e2col in [
            ('teff',  'st_teff',  True,  'st_tefferr1',  'st_tefferr2'),
            ('rad',   'st_rad',   True,  'st_raderr1',   'st_raderr2'),
            ('mass',  'st_mass',  True,  'st_masserr1',  'st_masserr2'),
            ('insol', 'pl_insol', True,  'pl_insolerr1', 'pl_insolerr2'),
            ('vmag',  'sy_vmag',  False, 'sy_vmagerr1',  'sy_vmagerr2'),
            ('dist',  'sy_dist',  True,  'sy_disterr1',  'sy_disterr2'),
            ('logg',  'st_logg',  True,  'st_loggerr1',  'st_loggerr2'),
            ('kmag',  'sy_kmag',  False, 'sy_kmagerr1',  'sy_kmagerr2'),
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
        # st_lum is log10(L/L_sun) in pscomppars/k2pandc — convert to linear L_sun.
        # Negative values are valid (sub-solar luminosity), so no require_positive.
        # Errors are in dex; convert each side through the exponential.
        if 'st_lum' in cols:
            log_lum = _safe_float(row['st_lum'])
            if log_lum is not None:
                lum = 10.0 ** log_lum
                entry['lum'] = lum
                pair = _safe_err_pair(
                    row['st_lumerr1'] if 'st_lumerr1' in cols else None,
                    row['st_lumerr2'] if 'st_lumerr2' in cols else None,
                )
                if pair is not None:
                    entry['lum_err1'] = 10.0 ** (log_lum + pair[0]) - lum
                    entry['lum_err2'] = lum - 10.0 ** (log_lum - pair[1])
        # metallicity: 0.0 is physically valid (solar), require_positive=False
        if 'st_met' in cols:
            met = _safe_float(row['st_met'])
            if met is not None:
                entry['met'] = met
                pair = _safe_err_pair(
                    row['st_meterr1'] if 'st_meterr1' in cols else None,
                    row['st_meterr2'] if 'st_meterr2' in cols else None,
                )
                if pair is not None:
                    entry['met_err1'], entry['met_err2'] = pair
        if 'pl_eqt' in cols:
            eqt = _safe_float(row['pl_eqt'], require_positive=True)
            if eqt is not None:
                entry['pl_eqt'] = eqt
                pair = _safe_err_pair(
                    row['pl_eqterr1'] if 'pl_eqterr1' in cols else None,
                    row['pl_eqterr2'] if 'pl_eqterr2' in cols else None,
                )
                if pair is not None:
                    entry['pl_eqt_err1'], entry['pl_eqt_err2'] = pair
        if 'st_spectype' in cols:
            spec = _safe_str(row['st_spectype'])
            if spec is not None:
                entry['spec'] = spec
        lookup[key] = entry
    return lookup


def _safe_float(val, require_positive: bool = False):
    """Return float value or None if absent/invalid/masked."""
    if np.ma.is_masked(val):
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    if require_positive and v <= 0:
        return None
    return v


def _safe_str(val):
    """Return stripped string or None if absent/sentinel."""
    s = str(val).strip()
    return None if s in ('', 'null', '--', 'nan') else s


def _safe_err_pair(e1, e2) -> tuple[float, float] | None:
    """Return (err1, err2) as positive magnitudes for asymmetric error propagation.

    NEA/EPIC/TOI convention: err1 ≥ 0 (upper), err2 ≤ 0 (lower).
    Both are returned as positive magnitudes so callers never need abs().
    If only one side is present the other is set to 0 (symmetric fallback).
    Returns None when both inputs are absent/non-finite.
    """
    v1 = _safe_float(e1)
    v2 = _safe_float(e2)
    if v1 is None and v2 is None:
        return None
    return (abs(v1) if v1 is not None else 0.0, abs(v2) if v2 is not None else 0.0)


class StellarParamSource:
    """Base class for a single tier in the stellar parameter merge chain.

    download → preprocess → load, pattern so each source supports file-based caching for offline use.

    Subclasses must implement download(), _build_lookup(), and declare key_col and source_name.
    Override get() when the lookup requires post-processing (e.g. derived rad).
    """

    key_col: str      # column in the catalog row used as lookup key
    source_name: str  # short label used in *_src provenance columns (e.g. "nea", "epic")

    def download(self, key_list: list[str]) -> Table:
        raise NotImplementedError

    def save_raw(self, key_list: list[str], path: str, format: str = "ascii") -> None:
        self.download(key_list).write(path, format=format, overwrite=True)

    def load_raw(self, path: str, format: str = "ascii") -> Table:
        return Table.read(path, format=format)

    def preprocess(self, raw: Table) -> Table:
        return raw

    def load(self, key_list: list[str] = None, from_file: str = None,
             format: str = "ascii") -> Table:
        """Download or read file, preprocess, and build the internal lookup dict."""
        raw = self.load_raw(from_file, format) if from_file else self.download(key_list or [])
        table = self.preprocess(raw)
        self._lookup = self._build_lookup(table)
        return table

    def _build_lookup(self, table: Table) -> dict:
        raise NotImplementedError

    def get(self, row) -> dict:
        """Return available stellar params for this catalog row, {} if no match."""
        key = str(row[self.key_col]).strip()
        return self._lookup.get(key, {})
