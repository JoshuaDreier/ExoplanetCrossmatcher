import astropy.units as u
from astropy.table import Table


class CatalogBase:
    """Abstract base class defining the interface for all catalog adapters.

    Subclasses provide source-specific download, preprocessing, and schema
    information.  The canonical usage pattern is::

        catalog = NEACatalog()
        table = catalog.load(from_file="pscomppars.txt")  # from cache
        # or
        table = catalog.load()  # downloads from remote TAP

    Attributes
    ----------
    ra_key : str
        Column name for right ascension in the catalog table.
    ra_unit : `~astropy.units.Unit`
        Unit of the ``ra_key`` column.
    dec_key : str
        Column name for declination in the catalog table.
    dec_unit : `~astropy.units.Unit`
        Unit of the ``dec_key`` column.
    host_key : str
        Column name identifying the host-star name.  Used as the join
        key in ID-based crossmatching.
    planet_uid : str
        Column that uniquely identifies a planet row (e.g. ``'pl_name'``
        for the NASA Exoplanet Archive).
    pm_key : str or None
        Total proper-motion column name, or ``None`` if the catalog does
        not carry proper-motion data.
    pmerr_key : str or None
        Proper-motion uncertainty column name, or ``None`` if not available.
    pm_unit : `~astropy.units.Unit`
        Astropy unit of the proper-motion columns.  Default ``u.mas/u.yr``.
        Used to convert pm values to arcsec/yr for the coordinate search
        radius calculation.
    """

    ra_key: str
    ra_unit: u.Unit
    dec_key: str
    dec_unit: u.Unit
    host_key: str
    planet_uid: str
    pm_key: str | None
    pmerr_key: str | None
    pm_unit: u.Unit = u.mas / u.yr

    def download(self) -> Table:
        """Query the remote data source and return the raw Table."""
        raise NotImplementedError

    def save_raw(self, path: str, format: str = "ascii", **kwargs) -> None:
        """Download raw data and write it to a file."""
        self.download().write(path, format=format, overwrite=True, **kwargs)

    def load_raw(self, path: str, format: str = "ascii", **kwargs) -> Table:
        """Read a raw file without preprocessing."""
        return Table.read(path, format=format, **kwargs)

    def preprocess(self, table: Table) -> Table:
        """Apply source-specific transformations to a raw catalog table.

        Default implementation is a no-op; subclasses override to add or
        transform columns (e.g. deriving ``coord_epoch`` for NEACatalog).

        Parameters
        ----------
        table : `~astropy.table.Table`
            Raw table returned by :meth:`download` or :meth:`load_raw`.

        Returns
        -------
        table : `~astropy.table.Table`
            Transformed table.  The concrete subclass documents which
            columns are added or modified.
        """
        return table

    def load(self, from_file: str | None = None, format: str = "ascii", **kwargs) -> Table:
        """Load the catalog and apply source-specific preprocessing.

        Parameters
        ----------
        from_file : str, optional
            Path to a previously saved raw file.  If given, reads from
            disk (no network call); otherwise calls :meth:`download`.
        format : str, optional
            Astropy table format string (default ``'ascii'``).
        **kwargs
            Forwarded to :meth:`load_raw` or :meth:`download`.

        Returns
        -------
        table : `~astropy.table.Table`
            Preprocessed catalog table.  Column schema depends on the
            concrete subclass; see :meth:`preprocess`.
        """
        raw = self.load_raw(from_file, format=format, **kwargs) if from_file \
              else self.download()
        return self.preprocess(raw)
