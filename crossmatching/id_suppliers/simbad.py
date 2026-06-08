import numpy as np
import pyvo
from astropy.table import Table

from crossmatching.id_suppliers.base import IdSupplierBase


class SimbadIdSupplier(IdSupplierBase):
    def _expand_id_with_variants(self, input_id, id_str):
        id_str = id_str.strip()
        variants = [id_str]

        if id_str.startswith('*'):
            # Strip SIMBAD object type prefix: * = Star, ** = Star in double system (binary)
            # Per SIMBAD nomenclature, these prefixes appear in ~4800 IDs but NEA catalog doesn't use them
            variants.append(id_str.lstrip('* '))
        if id_str.endswith("'s"):
            # SIMBAD sometimes includes possessive forms of star names (Barnard's), but not the non-possessive form
            # we include the non-possessive form, because NEA uses it (Barnard b)
            variants.append(id_str.rstrip("'s "))
        if id_str.startswith("NAME "):
            # SIMBAD sometimes includes "NAME " in front of star names (e.g. NAME Proxima Cen)
            # we include both versions
            variants.append(id_str.lstrip("NAME "))
        if len(id_str) > 2 and id_str[-2] == ' ' and id_str[-1] in 'ABCSN':
            # Some catalogs append a stellar component letter (e.g. 'Kepler-1229 A') while
            # others use the bare host name ('Kepler-1229'); include both forms
            variants.append(id_str[:-2])

        return [(input_id, v) for v in variants]

    def download(self, name_list: list[str]) -> Table:
        simbad = pyvo.dal.TAPService("https://simbad.cds.unistra.fr/simbad/sim-tap")
        return simbad.run_sync(
            "SELECT input_ids, ids FROM input_ids LEFT JOIN ident ON input_ids.input_ids = ident.id LEFT JOIN ids USING(oidref)",
            uploads={"input_ids": Table({self.input_col: name_list})}
        ).to_table()

    def load_alternate_ids(self, name_list: list[str], from_file: str = None, format: str = "ascii") -> Table:
        raw = self.load_raw(from_file, format=format) if from_file else self.download(name_list)
        if from_file:
            raw = raw[np.isin(raw[self.input_col], name_list)]
        return self.preprocess(raw)

    def preprocess(self, raw: Table) -> Table:
        input_ids = []
        all_ids = []
        for row in raw:
            input_id = str(row[self.input_col])
            for id_str in str(row["ids"]).split("|"):
                stripped = id_str.strip()
                if not stripped or stripped == self.null_sentinel:
                    continue
                for _, variant_id in self._expand_id_with_variants(input_id, id_str):
                    input_ids.append(input_id)
                    all_ids.append(variant_id)
        return Table([input_ids, all_ids], names=[self.input_col, self.id_col], dtype=["str", "str"])
