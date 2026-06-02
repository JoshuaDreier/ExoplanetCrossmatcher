from astropy.table import Table

from crossmatching.config import id_supplier as _cfg


class IdSupplierBase:
    input_col:     str = _cfg["input_col"]
    id_col:        str = _cfg["id_col"]
    null_sentinel: str = _cfg["null_sentinel"]

    def download(self, name_list: list[str]) -> Table:
        """Query the remote source and return the raw Table."""
        raise NotImplementedError

    def save_raw(self, name_list: list[str], path: str, format: str = "ascii") -> None:
        """Download raw data and write it to a file."""
        self.download(name_list).write(path, format=format, overwrite=True)

    def load_raw(self, path: str, format: str = "ascii") -> Table:
        """Read a raw file without preprocessing."""
        return Table.read(path, format=format)

    def preprocess(self, raw: Table) -> Table:
        """Apply source-specific transformations. Default: identity."""
        return raw

    def load_alternate_ids(self, name_list: list[str], from_file: str = None, format: str = "ascii") -> Table:
        """Load IDs, applying preprocessing. Reads from file if given, otherwise downloads."""
        raw = self.load_raw(from_file, format=format) if from_file else self.download(name_list)
        return self.preprocess(raw)
