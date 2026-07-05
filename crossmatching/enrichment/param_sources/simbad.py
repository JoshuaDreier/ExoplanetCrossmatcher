import pyvo
from astropy.table import Table
import numpy as np
import astropy.units as u

from crossmatching.enrichment.param_sources.base import ParamSource, _build_nea_style_lookup, _safe_float


class SimbadParamSource(ParamSource):
    """Stellar params from SIMBAD TAP (fallback tier for EMC-only planets).

    Queries basic + mesFe_h + allfluxes for stellar params, then a second
    query joins mesDiameter to retrieve direct stellar radii.
    Lookup key is the EMC 'main_id' column.
    Provides: teff, logg, met, spec, dist, vmag, kmag, rad (when mesDiameter
    has a km or mas entry).
    Note: cached ./input/simbad_params.txt files without 'logg', 'fe_h', or
    diameter columns are still readable (missing fields are silently skipped).
    Re-download to get the full parameter set.
    """
    key_col = "main_id"

    key_col = "main_id"
    source_name = "simbad"

    param_columns = {
        'teff': 'teff',
        'rad':  'simbad_rad',
        'spec': 'sp_type',
        'vmag': 'vmag',
        'logg': 'logg',
        'kmag': 'kmag',
        'met': 'fe_h',
        'dist': 'dist',
    }

    param_error_columns = {
        'teff': ('teff_err1', 'teff_err2'),
        'rad':  ('simbad_raderr1', 'simbad_raderr2'),
        'dist': ('dist_err1', 'dist_err2'),
    }

    def download(self, key_list: list[str]) -> Table:
        simbad = pyvo.dal.TAPService("https://simbad.cds.unistra.fr/simbad/sim-tap")
        upload = Table({"main_id": list(key_list)})

        main = simbad.run_sync(
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

        # Second query: one row per diameter measurement (unit is 'km  ' or 'mas ')
        diam = simbad.run_sync(
            """SELECT b.main_id, d.diameter, d.error, d.unit
               FROM TAP_UPLOAD.ids AS u
               JOIN basic        AS b ON b.main_id = u.main_id
               JOIN mesDiameter  AS d ON d.oidref  = b.oid""",
            uploads={"ids": upload},
        ).to_table()

        main.meta["_diameter_rows"] = diam
        return main

    def preprocess(self, raw: Table) -> Table:
        """Add derived columns from parallax and mesDiameter.

        1. Converts ``plx_value`` (mas) -> ``dist`` (pc) with error propagation.
        2. Aggregates ``mesDiameter`` rows (attached via ``raw.meta``) into
           ``simbad_rad`` (solar radii).  Two unit families are supported:

           * ``km``  - physical diameter; R = (diam/2) * km->R_sun
           * ``mas`` - angular diameter;  R = theta_rad * dist_pc * pc->R_sun
        """
        # ------------------------------------------------------------------
        # 1. Parallax -> distance
        # ------------------------------------------------------------------
        if 'plx_value' in raw.colnames:
            plx = raw['plx_value']
            raw['dist'] = np.where(plx > 0, 1000.0 / plx, np.nan)
            if 'plx_err' in raw.colnames:
                err = raw['plx_err']
                dist_err = np.where(plx > 0, 1000.0 * err / (plx ** 2), np.nan)
                raw['dist_err1'] = dist_err
                raw['dist_err2'] = dist_err

        # ------------------------------------------------------------------
        # 2. mesDiameter rows -> simbad_rad (solar radii)
        # ------------------------------------------------------------------
        diam_rows: Table | None = raw.meta.pop("_diameter_rows", None)
        if diam_rows is not None and len(diam_rows) > 0:
            # Build dist_pc lookup (needed for mas conversion)
            dist_by_id: dict[str, float] = {}
            if 'dist' in raw.colnames:
                for row in raw:
                    mid = str(row['main_id']).strip()
                    d = _safe_float(row['dist'])
                    if d is not None and d > 0:
                        dist_by_id[mid] = d

            # Collect per-star radius values; prefer km, accept mas as fallback
            from collections import defaultdict
            km_vals:  dict[str, list[float]] = defaultdict(list)
            km_errs:  dict[str, list[float]] = defaultdict(list)
            mas_vals: dict[str, list[float]] = defaultdict(list)
            mas_errs: dict[str, list[float]] = defaultdict(list)

            for drow in diam_rows:
                mid  = str(drow['main_id']).strip()
                unit = str(drow['unit']).strip().lower()
                diam = _safe_float(drow['diameter'])
                err  = _safe_float(drow['error'])
                if diam is None or diam <= 0:
                    continue

                if unit == 'km':
                    rad = (diam / 2.0) * u.km.to(u.R_sun)
                    km_vals[mid].append(rad)
                    if err is not None and err > 0:
                        km_errs[mid].append((err / 2.0) * u.km.to(u.R_sun))
                elif unit == 'mas':
                    dist_pc = dist_by_id.get(mid)
                    if dist_pc is None:
                        continue
                    theta_rad = (diam / 2.0) * 1e-3 * (np.pi / 648_000.0)
                    rad = theta_rad * dist_pc * u.pc.to(u.R_sun)
                    mas_vals[mid].append(rad)
                    if err is not None and err > 0:
                        theta_err = (err / 2.0) * 1e-3 * (np.pi / 648_000.0)
                        mas_errs[mid].append(theta_err * dist_pc * u.pc.to(u.R_sun))

            simbad_rad    = np.full(len(raw), np.nan)
            simbad_raderr = np.full(len(raw), np.nan)
            for i, row in enumerate(raw):
                mid = str(row['main_id']).strip()
                # Prefer km; fall back to mas
                vals = km_vals.get(mid) or mas_vals.get(mid)
                errs = km_errs.get(mid) or mas_errs.get(mid)
                if vals:
                    simbad_rad[i] = float(np.median(vals))
                    if errs:
                        simbad_raderr[i] = float(np.median(errs))

            raw['simbad_rad']     = simbad_rad
            raw['simbad_raderr1'] = simbad_raderr
            raw['simbad_raderr2'] = simbad_raderr

        return raw

    def _build_lookup(self, table: Table) -> dict:
        """Build lookup using unified helper after preprocessing.

        ``main_id`` is the key column, which matches the default ``key_col``.
        Calls ``preprocess`` first so the ``plx_value`` -> ``dist`` and
        ``mesDiameter`` -> ``simbad_rad`` conversions are applied even when
        ``_build_lookup`` is called directly (e.g. in tests).
        """
        table = self.preprocess(table)
        return _build_nea_style_lookup(table, self.param_columns, self.param_error_columns, key_col=self.key_col)
    # ms_radius is used as a fallback when simbad_rad is absent;
    # ParamFiller records provenance like "ms(teff:simbad)" in that case.
