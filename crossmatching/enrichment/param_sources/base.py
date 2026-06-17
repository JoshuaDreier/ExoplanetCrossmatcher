import numpy as np
from astropy.table import Table

from crossmatching.id_suppliers.base import IdSupplierBase


def _normalize_id_string(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    return " ".join(s.split())


def _build_nea_style_lookup(
    table: Table,
    columns: dict[str, str],
    errors: dict[str, tuple[str, str]],
    key_col: str = 'pl_name',
) -> dict:
    """Build a parameter lookup from an NEA‑style table.

    This version uses a unified CONFIG list that describes how each field
    should be read (float vs. string), whether it must be positive, and whether
    it has an associated error column.  Special cases such as luminosity
    conversion (log L → L) are handled explicitly after the generic loop.
    """
    # ---------------------------------------------------------------------
    # Unified configuration for all supported fields.
    # Each entry describes:
    #   * ``field`` – name used in the output ``entry`` dict.
    #   * ``positive`` – whether the value must be > 0 (passed to _safe_float).
    #   * ``has_error`` – whether the field may have asymmetric error columns.
    #   * ``type`` – "float" (default) or "str" for string-valued columns.
    #   * ``special`` – identifier for fields that need extra processing.
    # ---------------------------------------------------------------------
    CONFIG = [
        {"field": "teff",   "positive": True,  "has_error": True},
        {"field": "rad",    "positive": True,  "has_error": True},
        {"field": "mass",   "positive": True,  "has_error": True},
        {"field": "insol",  "positive": True,  "has_error": True},
        {"field": "vmag",   "positive": False, "has_error": True},
        {"field": "dist",   "positive": True,  "has_error": True},
        {"field": "logg",   "positive": True,  "has_error": True},
        {"field": "kmag",   "positive": False, "has_error": True},
        {"field": "lum",    "positive": False, "has_error": True, "special": "lum"},
        {"field": "met",    "positive": False, "has_error": True},
        {"field": "pl_eqt", "positive": True,  "has_error": True, "special": "pl_eqt"},
        {"field": "spec",   "positive": False, "has_error": False, "type": "str"},
    ]

    lookup: dict = {}
    cols = set(table.colnames)
    for row in table:
        key = _normalize_id_string(row[key_col])
        if not key or key in lookup:
            continue
        entry: dict = {}
        for cfg in CONFIG:
            field = cfg["field"]
            col_name = columns.get(field)
            if not col_name or col_name not in cols:
                continue
            # -----------------------------------------------------------------
            # Normal (non‑special) handling
            # -----------------------------------------------------------------
            if cfg.get("type") == "str":
                value = _safe_str(row[col_name])
            else:
                value = _safe_float(row[col_name], require_positive=cfg.get("positive", False))
            if value is None:
                continue
            # Store the primary value
            entry[field] = value
            # Errors – only for fields that declare ``has_error``
            if cfg.get("has_error"):
                err_cols = errors.get(field)
                if err_cols:
                    pair = _safe_err_pair(
                        row[err_cols[0]] if err_cols[0] in cols else None,
                        row[err_cols[1]] if err_cols[1] in cols else None,
                    )
                    if pair is not None:
                        entry[f"{field}_err1"], entry[f"{field}_err2"] = pair
            # -----------------------------------------------------------------
            # Special cases – luminosity conversion and equilibrium temperature
            # -----------------------------------------------------------------
            if cfg.get("special") == "lum":
                # ``value`` is the log‑luminosity; convert to linear units
                log_lum = value
                lum = 10.0 ** log_lum
                entry["lum"] = lum
                # Re‑compute asymmetric errors in linear space if available
                err_cols = errors.get("lum")
                if err_cols:
                    pair = _safe_err_pair(
                        row[err_cols[0]] if err_cols[0] in cols else None,
                        row[err_cols[1]] if err_cols[1] in cols else None,
                    )
                    if pair is not None:
                        entry["lum_err1"] = 10.0 ** (log_lum + pair[0]) - lum
                        entry["lum_err2"] = lum - 10.0 ** (log_lum - pair[1])
            elif cfg.get("special") == "pl_eqt":
                # ``value`` already respects ``positive`` flag; just store.
                entry["pl_eqt"] = value
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


class ParamSource:
    """Base class for a single tier in the stellar parameter merge chain.

    download → preprocess → load, pattern so each source supports file-based caching for offline use.

    Subclasses must implement download(), _build_lookup(), and declare key_col and source_name.
    Override get() when the lookup requires post-processing (e.g. derived rad).
    """

    key_col: str      # column in the catalog row used as lookup key
    source_name: str  # short label used in *_src provenance columns (e.g. "nea", "epic")

    param_columns: dict[str, str] = {}
    param_error_columns: dict[str, tuple[str, str]] = {}

    def _row_has(self, row, col: str) -> bool:
        if hasattr(row, 'colnames'):
            return col in row.colnames
        return col in row

    def _row_get(self, row, col: str):
        if hasattr(row, 'colnames'):
            return row[col] if col in row.colnames else None
        return row.get(col)

    def _normalize_key(self, key):
        return _normalize_id_string(key)

    def _lookup_key(self, row):
        if not self._row_has(row, self.key_col):
            return None
        return self._normalize_key(self._row_get(row, self.key_col))

    def _alternate_keys(self, row, input_starname_key: str, id_supplier: IdSupplierBase, alternate_ids: Table):
        if not self._row_has(row, input_starname_key):
            return []
        input_name = self._normalize_key(self._row_get(row, input_starname_key))
        if not input_name:
            return []
        if id_supplier.input_col not in alternate_ids.colnames or id_supplier.id_col not in alternate_ids.colnames:
            return []

        if not hasattr(alternate_ids, '_input_to_ids_map'):
            mapping = {}
            input_col = id_supplier.input_col
            id_col = id_supplier.id_col
            for alt_row in alternate_ids:
                alt_input = self._normalize_key(alt_row[input_col])
                alt_id = self._normalize_key(alt_row[id_col])
                if alt_input and alt_id:
                    mapping.setdefault(alt_input, []).append(alt_id)
            alternate_ids._input_to_ids_map = mapping

        id_candidates = alternate_ids._input_to_ids_map.get(input_name, [])

        keys: list[str] = [input_name]
        for alt_id in id_candidates:
            for variant in id_supplier.id_variants(alt_id):
                normalized = self._normalize_key(variant)
                if normalized and normalized not in keys:
                    keys.append(normalized)
        return keys

    def _get_float(self, row, field: str, require_positive: bool = False):
        col = self.param_columns.get(field)
        if not col or not self._row_has(row, col):
            return None
        return _safe_float(self._row_get(row, col), require_positive=require_positive)

    def _get_str(self, row, field: str):
        col = self.param_columns.get(field)
        if not col or not self._row_has(row, col):
            return None
        return _safe_str(self._row_get(row, col))

    def _get_err_pair(self, row, field: str):
        err_cols = self.param_error_columns.get(field)
        if not err_cols:
            return None
        return _safe_err_pair(
            self._row_get(row, err_cols[0]) if self._row_has(row, err_cols[0]) else None,
            self._row_get(row, err_cols[1]) if self._row_has(row, err_cols[1]) else None,
        )

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

    def get(
        self,
        row,
        input_starname_key: str | None = None,
        id_supplier: IdSupplierBase | None = None,
        alternate_ids: Table | None = None,
    ) -> dict:
        """Return available stellar params for this catalog row, {} if no match."""
        direct_key = self._lookup_key(row)
        if direct_key is not None and direct_key in self._lookup:
            return self._lookup[direct_key]

        if input_starname_key is None or id_supplier is None or alternate_ids is None:
            return {}

        keys = self._alternate_keys(row, input_starname_key, id_supplier, alternate_ids)
        for key in keys:
            if key in self._lookup:
                return self._lookup[key]
        return {}
