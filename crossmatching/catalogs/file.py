from astropy.table import Table

from crossmatching.catalogs.base import CatalogBase


class FileCatalog(CatalogBase):
    """
    Catalog adapter that reads directly from a file with no preprocessing.
    Useful for custom tables or cached pre-processed tables
    """

    def __init__(
        self,
        path: str,
        *,
        ra_key: str,
        dec_key: str,
        host_key: str,
        planet_uuid: str,
        pm_key: str = None,
        pmerr_key: str = None,
        format: str = "ascii",
    ):
        self.path = path
        self.format = format
        self.ra_key = ra_key
        self.dec_key = dec_key
        self.host_key = host_key
        self.planet_uuid = planet_uuid
        self.pm_key = pm_key
        self.pmerr_key = pmerr_key

    def download(self) -> Table:
        raise NotImplementedError("FileCatalog has no remote source. Use load_raw() directly.")

    def load(self, from_file: str = None, format: str = None, **kwargs) -> Table:
        path = from_file or self.path
        fmt = format or self.format
        return self.load_raw(path, format=fmt, **kwargs)
