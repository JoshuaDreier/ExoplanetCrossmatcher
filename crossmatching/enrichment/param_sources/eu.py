import pyvo
from astropy.table import Table
from crossmatching.enrichment.param_sources.base import ParamSource, _build_nea_style_lookup



class EuParamSource(ParamSource):
    # Column name in the local catalog used as the lookup key
    key_col = "name"
    source_name = "eu"

    # Mapping from the generic field names used by the merge chain to the
    # column names present in the ``exoplanet.epn_core`` table.  The table
    # follows the same naming conventions as the NASA Exoplanet Archive for
    # stellar parameters, but we keep the mapping explicit to avoid runtime
    # failures if the schema differs.
    param_columns = {
        "teff": "star_teff",
        "rad": "star_radius",
        "mass": "star_mass",
        "dist": "star_distance",
        "met": "star_metallicity",
        "pl_eqt": "temp_calculated",
        "spec": "star_spec_type",
    }
    # note exoplanet.eu's "log_g" column actually means planetary surface gravity

    param_error_columns = {
        "dist": ("star_distance_error_min", "star_distance_error_max"),
    }

    def download(self, key_list: list[str] = None) -> Table:
        """Retrieve the full ``exoplanet.epn_core`` table via the VO TAP service.

        The ``key_list`` argument is ignored because the service does not support
        selective retrieval of rows without an ADQL ``WHERE`` clause.  Fetching the
        entire table keeps the implementation simple and mirrors the behaviour of
        the other parameter sources.
        """
        service = pyvo.dal.TAPService("http://voparis-tap-planeto.obspm.fr/tap")
        return service.run_sync("SELECT * FROM exoplanet.epn_core").to_table()

    def _build_lookup(self, table: Table) -> dict:
        """Build a lookup dictionary using the shared NEA style helper.

        The helper handles column existence checks, unit conversion, and error
        handling uniformly across all sources.
        """
        return _build_nea_style_lookup(table, self.param_columns, self.param_error_columns, key_col="name")
