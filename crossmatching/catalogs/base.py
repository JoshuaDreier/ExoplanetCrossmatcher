from astropy.table import Table


class CatalogBase:
    ra_key: str 
    dec_key: str 
    host_key: str
    planet_uuid: str 
    pm_key: str | None 
    pmerr_key: str | None 

    def download(self) -> Table:
        """Query the remote data source and return the raw Table."""
        raise NotImplementedError

    def save_raw(self, path: str, format = "ascii", **kwargs) -> None:
        """Download raw data and write it to a file."""
        self.download().write(path, format=format, overwrite=True, **kwargs)

    def load_raw(self, path: str, format = "ascii", **kwargs) -> Table:
        """Read a raw file without preprocessing."""
        return Table.read(path, format=format, **kwargs)

    def preprocess(self, table: Table) -> Table:
        """Apply source-specific transformations (e.g. add coord_epoch).
        Default is a no preprocessing, just returning the raw table.
        """
        return table

    def load(self, from_file: str = None, format = "ascii", **kwargs) -> Table:
        """Load the catalog, applying preprocessing.
        Reads from from_file (raw) if given, otherwise calls download().
        """
        raw = self.load_raw(from_file, format=format, **kwargs) if from_file \
              else self.download()
        return self.preprocess(raw)
