import numpy as np
import pyvo
import astropy.units as u
from astropy.table import Table
from typing import Literal

from crossmatching.catalogs.base import CatalogBase

StatusValue = Literal["CONFIRMED", "CANDIDATE", "CONTROVERSIAL", "FALSE POSITIVE", "PRELIMINARY"]

_DEFAULT_ALLOWED_STATUSES: list[StatusValue] = ["CONFIRMED", "CANDIDATE", "CONTROVERSIAL", "PRELIMINARY"]


class EMCCatalog(CatalogBase):
    """
    Exo-MerCat catalog from the Exo-MerCat TAP service or reads it from file.
    Filters out entries marked as "FALSE POSITIVE" in the 'status' row from the raw catalogue
    """
    ra_key = "main_id_ra"
    ra_unit = u.degree
    dec_key = "main_id_dec"
    dec_unit = u.degree
    host_key = "host"
    planet_uid = "exo-mercat_name"
    pm_key = None
    pmerr_key = None

    ENRICH_KEYS = {
        "planet_radius_key": "r",
        "semi_major_axis_key": "a",
        "period_key": "p",
        "msini_key": "msini",
    }

    def __init__(self, allowed_statuses: list[StatusValue] | None = None):
        self.allowed_statuses = allowed_statuses if allowed_statuses is not None else _DEFAULT_ALLOWED_STATUSES

    def download(self) -> Table:
        """
        !WARNING: at the time of writing, Exo-MerCat TAP service is not regularly updated,
        it is recommended to locally run Exo-MerCat and load the result as a file.
        """
        emc = pyvo.dal.TAPService("http://archives.ia2.inaf.it/vo/tap/projects/")
        return emc.run_sync("SELECT * FROM exomercat.exomercat").to_table()

    def preprocess(self, table: Table) -> Table:
        return table[np.isin(table["status"], self.allowed_statuses)]
