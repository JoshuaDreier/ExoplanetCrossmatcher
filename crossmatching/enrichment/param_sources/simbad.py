import pyvo
from astropy.table import Table
import numpy as np

from crossmatching.enrichment.param_sources.base import ParamSource, _safe_float, _safe_str, _build_nea_style_lookup


class SimbadParamSource(ParamSource):
    """Stellar params from SIMBAD TAP (fallback tier for EMC-only planets).

    Queries basic + mesFe_h + allfluxes for a batch of main_id strings.
    Lookup key is the EMC 'main_id' column.
    Provides: teff, logg, met, spec, dist, vmag, kmag; rad is derived via ms_radius_from_teff.
    Note: cached simbad_params.txt files without 'logg' or 'fe_h' columns are still
    readable (missing fields are silently skipped). Re-download to get metallicity.
    """
    key_col = "main_id"

    key_col = "main_id"
    source_name = "simbad"

    param_columns = {
        'teff': 'teff',
        'spec': 'sp_type',
        'vmag': 'vmag',
        'logg': 'logg',
        'kmag': 'kmag',
        'met': 'fe_h',
        'dist': 'dist',
    }

    param_error_columns = {
        'teff': ('teff_err1', 'teff_err2'),
        'dist': ('dist_err1', 'dist_err2'),
    }

    def download(self, key_list: list[str]) -> Table:
        simbad = pyvo.dal.TAPService("https://simbad.cds.unistra.fr/simbad/sim-tap")
        upload = Table({"main_id": list(key_list)})
        return simbad.run_sync(
            """SELECT b.main_id, b.sp_type, b.plx_value, b.plx_err,
                      AVG(f.teff) AS teff, AVG(f.log_g) AS logg, AVG(f.fe_h) AS fe_h,
                      MAX(fl.V) AS vmag, MAX(fl.K) AS kmag
               FROM TAP_UPLOAD.ids AS u
               JOIN basic AS b ON b.main_id = u.main_id
               LEFT JOIN mesFe_h   AS f  ON f.oidref = b.oid
               LEFT JOIN allfluxes AS fl ON fl.oidref = b.oid
               GROUP BY b.main_id, b.sp_type, b.plx_value, b.plx_err""",
            uploads={"ids": upload},
        ).to_table()

    def preprocess(self, raw: Table) -> Table:
        """Add derived ``dist`` column from parallax.

        The SIMBAD TAP service returns ``plx_value`` (parallax in mas) and
        ``plx_err``.  We convert to distance in parsec (``dist = 1000 / plx``) and
        propagate the error symmetrically.
        """
        if 'plx_value' in raw.colnames:
            # Avoid division by zero
            plx = raw['plx_value']
            raw['dist'] = np.where(plx > 0, 1000.0 / plx, np.nan)
            if 'plx_err' in raw.colnames:
                err = raw['plx_err']
                # error propagation for d = 1000 / p: sigma_d = 1000 * sigma_p / p^2
                dist_err = np.where(plx > 0, 1000.0 * err / (plx ** 2), np.nan)
                raw['dist_err1'] = dist_err
                raw['dist_err2'] = dist_err
        return raw

    def _build_lookup(self, table: Table) -> dict:
        """Build lookup using unified helper after preprocessing.

        ``main_id`` is the key column, which matches the default ``key_col``.
        Calls ``preprocess`` first so the ``plx_value`` → ``dist`` conversion
        is applied even when ``_build_lookup`` is called directly (e.g. in tests).
        """
        table = self.preprocess(table)
        return _build_nea_style_lookup(table, self.param_columns, self.param_error_columns, key_col=self.key_col)
    # ms_radius is intentionally NOT applied here; the ParamFiller fallback
    # handles it so that provenance (e.g. "ms(teff:simbad)") is recorded correctly.
