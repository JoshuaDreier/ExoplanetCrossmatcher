# Getting Started

## Prerequisites

- Python 3.10+
- Network access for first-time downloads (SIMBAD and NEA TAP endpoints)

## Install

```bash
pip install -r requirements.txt
```

Key dependencies: `astropy`, `pyvo`, `pandas`, `numpy`, `pytest`.

## Data files

The crossmatcher separates network calls from computation.  Download once and
cache; subsequent runs read from disk.

| File | Contents | How to obtain |
|------|----------|---------------|
| `./input/pscomppars.txt` | NASA Exoplanet Archive `pscomppars` table | `NEACatalog().save_raw("./input/pscomppars.txt")` |
| `./input/alternate_ids_hpic.txt` | SIMBAD alternate IDs for all HPIC stars | `SimbadIdSupplier().save_raw(name_list, "./input/alternate_ids_hpic.txt")` |
| `./input/exo-mercat.csv` | Exo-MerCat merged catalog | Download from the [Exo-MerCat releases](https://github.com/Exo-MerCat/Exo-MerCat) |
| `input/HPIC_LC4_combined_d50.txt` | Input stellar survey (pipe-delimited) | Provided in `input/` |

## First crossmatch — NASA Exoplanet Archive

This is the canonical workflow, matching HPIC LC4 against NEA using SIMBAD as
the identifier broker.

```python
from astropy.table import Table
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier

# Load input stellar survey
input_table = Table.read("input/HPIC_LC4_combined_d50.txt", format="ascii")

# Build the crossmatcher
cm = Crossmatcher(
    catalog=NEACatalog(),
    id_supplier=SimbadIdSupplier(),
)


# this line only needs to run once, or when newer catalog data is wanted
cm.catalog.download().write("./input/pscomppars.txt", overwrite=True, format="ascii") 
cm.load_catalog(from_file="./input/pscomppars.txt")


# cm.id_supplier.download(hpic["star_name"]).write("./input/alternate_ids_hpic.txt", overwrite=True, format="ascii")
# this line only needs to run once, or when catalog ids changed
cm.load_alternate_ids(input_table["star_name"], from_file="./input/alternate_ids_hpic.txt")

# Run the combined crossmatch (ID-based first, then coordinate-based)
result = cm.combined_crossmatch(input_table, input_starname_key="star_name")


from collections import Counter
print(Counter(result["match_type"])) # see the frequency of the used matching methods

result["star_name", "pl_name", "match_type", "crossmatching_angular_separation"]
# subtable is returned as cell output, alternatively use .show_in_notebook() or .show_in_browser() 
```

## First crossmatch — Exo-MerCat

Exo-MerCat bundles its own alternate IDs; no SIMBAD query is needed.  Both
catalog and ID supplier read from the same CSV file.

```python
from crossmatching import Crossmatcher, EMCCatalog, EMCIdSupplier

cme = Crossmatcher(EMCCatalog(), EMCIdSupplier())
cme.load_catalog(from_file="./input/exo-mercat.csv")
cme.load_alternate_ids(hpic["star_name"].tolist(), from_file="./input/exo-mercat.csv")

emc_result = cme.combined_crossmatch(hpic, input_starname_key="star_name")
```

## Reading the output

`combined_crossmatch` returns an `astropy.table.Table`.  Key columns:

| Column | Description |
|--------|-------------|
| `star_name` | Input star name from HPIC |
| `pl_name` | Planet identifier from the catalog |
| `match_type` | `'id'` — found via identifier; `'coordinates'` — found by sky position |
| `angular_separation` | Angular distance (arcsec) between input and catalog positions (coordinate matches only) |

All other columns pass through from the catalog (e.g. all `pscomppars` columns
for an NEA match).  See [Column Reference](column-reference.md) for the full schema.

```python
from collections import Counter

# How many matches came from each method?
print(Counter(result["match_type"]))
# → Counter({'id': 4832, 'coordinates': 61})

# Which planets were found?
print(result["pl_name", "match_type"])
```
