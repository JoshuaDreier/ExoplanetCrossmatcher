import re
import numpy as np
import pyvo
from astropy.table import Table, MaskedColumn

from crossmatching.catalogs.base import CatalogBase

class EMCCatalog(CatalogBase):
    """Exo-MerCat catalog from the Exo-MerCat TAP service."""
    ra_key = "main_id_ra"
    dec_key = "main_id_dec"
    host_key = "host"
    planet_uuid = "exo-mercat_name"
    pm_key = None
    pmerr_key = None

    def download(self) -> Table:
        """
        !WARNING: at the time of writing, Exo-MerCat TAP service is not regularly updates, 
        it is recommended to locally run Exo-MerCat and loading the result as a file.
        """
        # at the time of writing, exomercat TAP is n
        emc = pyvo.dal.TAPService("http://archives.ia2.inaf.it/vo/tap/projects/")
        return emc.run_sync("SELECT * FROM exomercat.exomercat").to_table()

    def preprocess(self, table: Table) -> Table:
        return table

