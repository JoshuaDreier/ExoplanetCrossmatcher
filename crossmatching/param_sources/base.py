import numpy as np
from astropy.table import Table


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
