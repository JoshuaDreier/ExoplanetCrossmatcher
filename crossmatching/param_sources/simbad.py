import pyvo
from astropy.table import Table

from crossmatching.param_sources.base import StellarParamSource, _safe_float, _safe_str


class SimbadStellarParamSource(StellarParamSource):
    """Stellar params from SIMBAD TAP (fallback tier for EMC-only planets).

    Queries basic + mesFe_h + allfluxes for a batch of main_id strings.
    Lookup key is the EMC 'main_id' column.
    Provides: teff, logg, met, spec, dist, vmag, kmag; rad is derived via ms_radius_from_teff.
    Note: cached simbad_params.txt files without 'logg' or 'fe_h' columns are still
    readable (missing fields are silently skipped). Re-download to get metallicity.
    """

    key_col = "main_id"
    source_name = "simbad"

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

    def _build_lookup(self, table: Table) -> dict:
        cols = set(table.colnames)
        lookup = {}
        for row in table:
            mid = str(row['main_id']).strip()
            if mid in lookup:
                continue
            entry = {}
            teff = _safe_float(row['teff'], require_positive=True)
            spec = _safe_str(row['sp_type'])
            plx  = _safe_float(row['plx_value'], require_positive=True)
            vmag = _safe_float(row['vmag'])
            if teff is not None:
                entry['teff'] = teff
            if spec is not None:
                entry['spec'] = spec
            if plx is not None:
                entry['dist'] = 1000.0 / plx
                if 'plx_err' in cols:
                    plx_err = _safe_float(row['plx_err'], require_positive=True)
                    if plx_err is not None:
                        # d = 1000/plx → σ_d = 1000 * σ_plx / plx²; symmetric (single plx_err)
                        d_err = 1000.0 * plx_err / plx ** 2
                        entry['dist_err1'] = d_err
                        entry['dist_err2'] = d_err
            if vmag is not None:
                entry['vmag'] = vmag
            if 'logg' in cols:
                logg = _safe_float(row['logg'], require_positive=True)
                if logg is not None:
                    entry['logg'] = logg
            if 'kmag' in cols:
                kmag_v = _safe_float(row['kmag'])
                if kmag_v is not None:
                    entry['kmag'] = kmag_v
            if 'fe_h' in cols:
                fe_h = _safe_float(row['fe_h'])  # 0.0 is valid (solar), no require_positive
                if fe_h is not None:
                    entry['met'] = fe_h
            lookup[mid] = entry
        return lookup

    # ms_radius is intentionally NOT applied here; the StellarParamMerger fallback
    # handles it so that provenance (e.g. "ms(teff:simbad)") is recorded correctly.
