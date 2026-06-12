import numpy as np
from astropy.table import Table

from crossmatching.id_suppliers.base import IdSupplierBase


class EMCIdSupplier(IdSupplierBase):
    """
    Derives alternate IDs directly from Exo-MerCat's main_id_aliases column,
    avoiding a SIMBAD query. Aliases are comma-separated SIMBAD IDs
    stored per host star.

    The alternate_ids table maps potential HPIC input names (alias variants)
    to the corresponding EMC host name, which is what appears in EMCCatalog's
    host column.
    """
    aliases_col = "main_id_aliases"
    host_col    = "host"

    def id_variants(self, id_str: str) -> list[str]:
        """Shared rules from IdSupplierBase plus two HPIC-specific spellings."""
        variants = super().id_variants(id_str)
        id_str = id_str.strip()
        if id_str.startswith("Gaia DR"):
            # EMC stores "Gaia DR3 ..." but HPIC uses "GAIA DR3 ..."
            variants.append("GAIA" + id_str[4:])
        if id_str.startswith("TIC-") and id_str[4:].isdigit():
            # EMC stores "TIC-XXXXXXX" but HPIC uses "TIC XXXXXXX"
            variants.append("TIC " + id_str[4:])
        return variants

    def preprocess(self, raw: Table) -> Table:
        input_ids = []
        all_ids = []
        seen_hosts = set()
        for row in raw:
            host = str(row[self.host_col])
            if host in seen_hosts:
                continue
            seen_hosts.add(host)
            aliases_str = str(row[self.aliases_col])
            if not aliases_str or aliases_str == self.null_sentinel:
                continue
            for alias in aliases_str.split(","):
                alias = " ".join(alias.split())
                if not alias or alias == self.null_sentinel:
                    continue
                # Alias variants become potential HPIC input names; host is the catalog key.
                for variant in self.id_variants(alias):
                    input_ids.append(variant)
                    all_ids.append(host)
        return Table([input_ids, all_ids], names=[self.input_col, self.id_col], dtype=["str", "str"])

    def download(self, name_list: list[str]) -> Table:
        raise NotImplementedError(
            "EMCIdSupplier reads from a local catalog file; use load_alternate_ids(from_file=...)"
        )

    def load_alternate_ids(self, name_list: list[str], from_file: str = None, format: str = "csv") -> Table:
        if not from_file:
            raise ValueError("EMCIdSupplier requires a catalog file path via from_file=")
        raw = self.load_raw(from_file, format=format)
        processed = self.preprocess(raw)
        return processed[np.isin(processed[self.input_col], name_list)]
