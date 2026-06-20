import pyvo
from astropy.table import Table

from crossmatching.enrichment.param_sources.base import ParamSource, _build_nea_style_lookup


class EpicParamSource(ParamSource):
    """Stellar and planetary params from the NASA K2 Candidates and Planets table (k2pandc).

    Covers K2/EPIC planets not present in the main pscomppars table (~68% unique population, 2026-08-06).
    Lookup key is the EMC 'epic_name' column, which maps to 'pl_name' in k2pandc.
    Provides: teff, rad, mass, spec, vmag, dist, insol, logg, met, pl_eqt.
    """

    key_col = "epic_name"
    source_name = "epic"

    param_columns = {
        'teff': 'st_teff',
        'rad': 'st_rad',
        'mass': 'st_mass',
        'insol': 'pl_insol',
        'vmag': 'sy_vmag',
        'dist': 'sy_dist',
        'logg': 'st_logg',
        'kmag': 'sy_kmag',
        'lum': 'st_lum',
        'met': 'st_met',
        'pl_eqt': 'pl_eqt',
        'pl_rad': 'pl_radj',
        'pl_mass': 'pl_massj',
        'msini': 'pl_msinij',
        'spec': 'st_spectype',
    }

    param_error_columns = {
        'teff': ('st_tefferr1', 'st_tefferr2'),
        'rad': ('st_raderr1', 'st_raderr2'),
        'mass': ('st_masserr1', 'st_masserr2'),
        'insol': ('pl_insolerr1', 'pl_insolerr2'),
        'vmag': ('sy_vmagerr1', 'sy_vmagerr2'),
        'dist': ('sy_disterr1', 'sy_disterr2'),
        'logg': ('st_loggerr1', 'st_loggerr2'),
        'kmag': ('sy_kmagerr1', 'sy_kmagerr2'),
        'lum': ('st_lumerr1', 'st_lumerr2'),
        'met': ('st_meterr1', 'st_meterr2'),
        'pl_eqt': ('pl_eqterr1', 'pl_eqterr2'),
        'pl_rad': ('pl_radjerr1', 'pl_radjerr2'),
        'pl_mass': ('pl_massjerr1', 'pl_massjerr2'),
        'pl_msini': ('pl_msinijerr1', 'pl_msinijerr2'),
    }

    def download(self, key_list: list[str] = None) -> Table:
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        return nasa.run_sync("SELECT * FROM k2pandc").to_table()

    def _build_lookup(self, table: Table) -> dict:
        """Build lookup dict for EPIC source using unified helper.

        The EPIC table uses ``pl_name`` as its identifier column, which matches the
        default ``key_col`` of ``_build_nea_style_lookup``.  We simply forward the
        column‑mapping dictionaries defined on the class.
        """
        return _build_nea_style_lookup(table, self.param_columns, self.param_error_columns)

