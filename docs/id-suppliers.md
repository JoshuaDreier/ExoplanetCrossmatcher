# ID Suppliers
An ID supplier maps input star names to alternate identifiers, enabling `id_crossmatch()` to join against catalog host-star names that use different conventions.  Both suppliers extend `IdSupplierBase` and share the same `load_alternate_ids(from_file=...)` caching interface.

## SimbadIdSupplier
Queries the SIMBAD TAP service for all known identifiers of each input star.

```python
from crossmatching import SimbadIdSupplier

supplier = SimbadIdSupplier()

# Download for a list of star names and save to disk (~1.5 minutes for 15 000 stars)
supplier.save_raw(name_list, "alternate_ids_hpic.txt")

# Load from cache on subsequent runs
ids = supplier.load_alternate_ids(name_list, from_file="alternate_ids_hpic.txt")
```
The raw result is a two-column table: `input_ids` (input name) and `ids` (pipe-delimited SIMBAD identifiers).
When loaded this is then expanded into more rows with each input_id and alias pair occupying one row.
### Variant expansion
For each raw identifier, `SimbadIdSupplier` generates up to four additional
variant forms:

| Rule                                | Input                | Output          | Reason                                         |
| ----------------------------------- | -------------------- | --------------- | ---------------------------------------------- |
| Strip `*` / `**` object-type prefix | `'* alf Cen'`        | `'alf Cen'`     | SIMBAD prepends object-type codes; NEA doesn't |
| Strip `'s` possessive suffix        | `"Barnard's"`        | `'Barnard'`     | NEA uses `'Barnard b'`, not `"Barnard's b"`    |
| Strip `NAME ` prefix                | `'NAME Proxima Cen'` | `'Proxima Cen'` | SIMBAD `NAME` tag is catalog-internal          |
| Drop trailing component letter      | `'Kepler-1229 A'`    | `'Kepler-1229'` | Some catalogs append `A`/`B`/`C`/`S`/`N`       |

These variants bridge the naming gap between SIMBAD's internal identifiers and the host-name column in the planet catalog.

The cache file may contain IDs for more stars than a given run needs. `SimbadIdSupplier.load_alternate_ids()` automatically subsets the file to the requested `name_list` before returning, so you can maintain one large cache and
query subsets cheaply.



## EMCIdSupplier
Derives alternate IDs directly from Exo-MerCat's `main_id_aliases` column.  No SIMBAD query is needed, but internally Exo-MerCat also uses SIMBAD with its own rules for adding more aliases; both catalog and ID supplier read from the same file.

```python
from crossmatching import EMCIdSupplier

supplier = EMCIdSupplier()
ids = supplier.load_alternate_ids(name_list, from_file="exo-mercat.csv")
```

`EMCIdSupplier` has no `download()` method; `from_file=` is required.

### Alias expansion
EMCIdSupplier applies a similar variant-expansion set to `SimbadIdSupplier`,
with two additional rules specific to naming differences between EMC and HPIC:

| Rule                   | Input            | Output           |
| ---------------------- | ---------------- | ---------------- |
| Capitalise Gaia prefix | `'Gaia DR3 ...'` | `'GAIA DR3 ...'` |
| TIC hyphen → space     | `'TIC-12345678'` | `'TIC 12345678'` |

### Join direction
The mapping direction is reversed relative to `SimbadIdSupplier`: the alias variants become the potential **input** names, and the EMC `host` value is the **catalog key**.  This lets `id_crossmatch()` find the correct EMC row even when
the input_table uses a non-canonical form of the star name i.e. not corresponding to `main_id`).


## Implementing a custom ID supplier
Subclass `IdSupplierBase` and implement `download()` and `preprocess()`.
```python
from astropy.table import Table
from crossmatching.id_suppliers.base import IdSupplierBase

class MyIdSupplier(IdSupplierBase):
    def download(self, name_list: list[str]) -> Table:
        # Query your ID source here
        ...

    def preprocess(self, raw: Table) -> Table:
        # Return a two-column Table: input_col, id_col
        ...
```
