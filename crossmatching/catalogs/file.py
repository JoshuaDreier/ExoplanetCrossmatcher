import astropy.units as u
from astropy.table import Table
from crossmatching.catalogs.base import CatalogBase


class FileCatalog(CatalogBase):
    """Catalog adapter that reads directly from a local file.

    Useful for custom planet tables or for pre-processed catalog files
    saved with :meth:`~CatalogBase.save_raw`.  No preprocessing is
    applied; the file is read as-is.

    Parameters
    ----------
    path : str
        Default path to the catalog file.
    ra_key : str
        Column name for right ascension.
    ra_unit : `~astropy.units.Unit`
        Unit of the ``ra_key`` column.
    dec_key : str
        Column name for declination.
    dec_unit : `~astropy.units.Unit`
        Unit of the ``dec_key`` column.
    host_key : str
        Column name for host-star names (join key for ID matching).
    planet_uuid : str
        Column that uniquely identifies each planet row.
    pm_key : str, optional
        Column name for total proper motion.  ``None`` if the file does
        not contain proper-motion data.
    pmerr_key : str, optional
        Column name for proper-motion uncertainty.  ``None`` if not available.
    pm_unit : `~astropy.units.Unit`, optional
        Unit of the proper-motion columns.  Default ``u.mas/u.yr``.
    format : str, optional
        Astropy table format string for reading the file
        (default ``'ascii'``).
    """

    def __init__(
        self,
        path: str,
        *,
        ra_key: str,
        ra_unit: u.Unit,
        dec_key: str,
        dec_unit: u.Unit,
        host_key: str,
        planet_uuid: str,
        pm_key: str | None = None,
        pmerr_key: str | None = None,
        pm_unit: u.Unit = u.mas / u.yr,
        format: str = "ascii",
    ):
        self.path = path
        self.format = format
        self.ra_key = ra_key
        self.ra_unit = ra_unit
        self.dec_key = dec_key
        self.dec_unit = dec_unit
        self.host_key = host_key
        self.planet_uuid = planet_uuid
        self.pm_key = pm_key
        self.pmerr_key = pmerr_key
        self.pm_unit = pm_unit

    def download(self) -> Table:
        raise NotImplementedError("FileCatalog has no remote source. Use load_raw() directly.")

    def load(self, from_file: str = None, format: str = None, **kwargs) -> Table:
        """Read the catalog file and return the table without preprocessing.

        Parameters
        ----------
        from_file : str, optional
            Override path.  If given, reads this file instead of the
            path supplied to the constructor.
        format : str, optional
            Override file format.  Falls back to the constructor
            ``format`` argument if not given.
        **kwargs
            Forwarded to :meth:`~CatalogBase.load_raw`.

        Returns
        -------
        table : `~astropy.table.Table`
            File contents with no preprocessing applied.
        """
        path = from_file or self.path
        fmt = format or self.format
        return self.load_raw(path, format=fmt, **kwargs)
