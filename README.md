[![astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](https://www.astropy.org/)
[![Python](https://img.shields.io/badge/python-3.14-blue?style=flat&logo=python)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=flat&logo=pytest)](https://docs.pytest.org/)
[![Jupyter](https://img.shields.io/badge/best%20used%20with-Jupyter-F37626?style=flat&logo=jupyter)](https://jupyter.org/)

# Exoplanet Crossmatcher

A Python library that identifies exoplanet host stars by matching an input stellar survey against exoplanet catalogs using two complementary strategies:
- identifier-based matching
- coordinate matching

Crossmatch results can be enriched with stellar and derived planetary parameters from multiple sources (HPIC, NASA Exoplanet Archive (abbr. NEA), SIMBAD, K2 EPIC, TESS TOI).

## Features
- Match any list of input stars (with just names or names + coordinates) against the NASA Exoplanet Archive, Exo-MerCat, or other catalogs.
- SIMBAD alternate-ID resolution handles some inconsistent catalog spellings
- Per-row coordinate search radius grows with proper motion to avoid epoch mismatches (only applies to )
- File-based caching, function to download once, then run offline 
- Priority-ordered stellar parameter merging from a variety of catalogs with asymmetric error propagation
- Configurable Rocky and temperate planet classification masks

## Installation

After cloning the repository, create a virtual environment (example with `venv`)
```
python3 -m venv ".venv"
source .venv/bin/activate  # Activate the virtual environment (bash) 
# .venv/Scripts/activate for windows 
```
then execute the following in the project directory:
```bash
pip install -r requirements.txt
```

The project was built and tested in Python 3.14.


## Quick start
The optimal place to work with this library are Jupyter Notebooks
### Crossmatching with NASA Exoplanet Archive

```python
from astropy.table import Table
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier

# Load input stellar survey
input_table = Table.read("./input/HPIC_LC4_combined_d50.txt", format="ascii")
# this can be any astropy table

# Build the crossmatcher
cm = Crossmatcher(
    catalog=NEACatalog(),
    id_supplier=SimbadIdSupplier(),
    minimum_search_radius = 10*u.arcsec,
    default_search_radius = 50*u.arcsec,
    input_suffix = "input",
    input_epoch = 2000 # depends on coordinate source for input_table, for HPIC it's 2000
)



cm.catalog.save_raw("./input/pscomppars.txt") # this line only needs to run once, or when newer catalog data is wanted
cm.load_catalog(from_file="./input/pscomppars.txt")


cm.id_supplier.save_raw(hpic["star_name"], "./input/alternate_ids_hpic.txt") # this line only needs to run once, or when catalog ids changed
cm.load_alternate_ids(input_table["star_name"], from_file="./input/alternate_ids_hpic.txt")

# Run the combined crossmatch (ID-based first, then coordinate-based)
result = cm.combined_crossmatch(input_table, input_starname_key="star_name")


# viewing the results

from collections import Counter
print(Counter(result["match_type"])) # see the occurence of the used matching methods

result["star_name", "pl_name", "match_type", "crossmatching_angular_separation"] # cell output, alternatively use .show_in_notebook() or .show_in_browser() 
```
Expect this to run for ~3 minutes on the first run due to the NASA & SIMBAD TAP queries (each taking ~1.5 min) On second run, with commented out .download line, the code should run in seconds. If `from_file=...`is left out, the download will be performed too (but is not saved to disk).

The `match_type` column is 
- `'id'` for identifier matches,
- `'coordinates'` for position-based matches, and
- `'id+coordinates'` if a planet is matched via both methods.

If a collision between column names between the input catalog and the `cm.catalog`occurs, the input catalog's columns are suffixed with `_input`and all catalog columns pass through unchanged.

In case only id-matching or coordinate crossmatching is desired, replace `cm.combined_crossmatch(...)` with
```python
result = cm.id_crossmatch(input_table, input_starname_key="star_name")
``` 
(does not require coordinates but, when provided and keyword arguments ra_key, dec_key are given will compute angular separation)
Conversely, if only a coordinate crossmatch is wanted, use
```python
result = cm.coordinate_crossmatch(input_table, ra_key="ra", dec_key="dec") # those keyword arguments are the default
```

### Crossmatching with Exo-MerCat
The code from above just changes in the used classes and loaded file names.
Note that at time of writing the [Exo-MerCat TAP](https://exo-mercat.readthedocs.io/en/latest/run_tap.html) service is not kept up to date (and missing the alias column), so ideally clone [Exo-MerCat](https://github.com/Exo-MerCat/Exo-MerCat), and copy the resulting `exomarcat.csv`file to this directory. A timestamped version is provided in this projects `input/` directory.
Note also that exomercat provides no information on proper motion. so default_search_radius will be used for all coordinate matched.

```python
from astropy.table import Table
import astropy.units as u
from crossmatching import EMCCatalog, EMCIdSupplier, Crossmatcher

catalog = EMCCatalog()
# note, this packages implementation of EMCCatalog removes planet rows flagged as "FALSE POSITIVE" 
# this can be changed by instatiating 
# catalog = EMCCatalog(allowed_statuses=["CONFIRMED", "CANDIDATE", "CONTROVERSIAL", "FALSE POSITIVE", "PRELIMINARY"])

cme = Crossmatcher(catalog, EMCIdSupplier(), default_search_radius=50*u.arcsec)
cme.load_catalog(from_file="./input/exo-mercat.csv")

cme.load_alternate_ids(input_table["star_name"], from_file="./input/exo-mercat.csv")

emc_result = cme.combined_crossmatch(input_table, input_starname_key="star_name")
```

### Implementing your own catalog
[`crossmatching/catalogs/base.py`](crossmatching/catalogs/base.py) Contains the `CatalogBase` class, which is the type of the `catalog` argument in `Crossmatcher`. By extending the class as follows, any TAP catalog or file can be fed as a source to  `Crossmatcher`. 
```python
from crossmatching.catalogs.base import CatalogBase

class MyTAPCatalog(CatalogBase):
    ra_key = "ra"
    dec_key = "dec"
    hostname_key = "host"       
    planet_uid = "planet_name" 
    pm_key = None               
    pmerr_key = None            

    def download(self) -> Table:
        service = pyvo.dal.TAPService("http://my-tap-service.example.org/tap")
        return service.search("SELECT * FROM my_schema.my_planets").to_table()

    def preprocess(self, table: Table) -> Table:g, like 
	    # optional preprocessing, like deleting certain rows
        return table    
``` 
When `load_catalog` is called, `download`is run, then `preprocess` is applied on its result. If a  `from_file` kwarg is present, then the given file (parsed as a `astropy.table.Table`) is fed to `preprocess`. Then replace `NEACatalog` in the code above with the new class.

## Documentation
The full user documentation lives in [`docs/`](docs/):

| Page                                         | Contents                                              |
| -------------------------------------------- | ----------------------------------------------------- |
| [Crossmatching](docs/crossmatching.md)       | ID matching, coordinate matching, combined strategy   |
| [Column Reference](docs/column-reference.md) | Full output column schema with units                  |
| [Architecture](docs/architecture.md)         | Package layout, data-flow                             |
| [Catalogs](docs/catalogs.md)                 | `NEACatalog`, `FileCatalog`, `EMCCatalog`             |
| [ID Suppliers](docs/id-suppliers.md)         | `SimbadIdSupplier`, `EMCIdSupplier`                   |
| [Enrichment](docs/enrichment.md)             | `ParamFiller`, parameter sources, derived columns, candidate masks |
| [Derived Parameter Inference](docs/derived-parameter-inference.md) | Formulas and citations for enrichment fallback calculations |
| [Detect Duplicates](detect-duplicates.md)    | Detection and deletion of duplicates based on aliases |
| [Configuration](docs/configuration.md)       | `crossmatching.cfg` keys and override patterns        |
| [Testing](docs/testing.md)                   | Types of tests, how to run                            |

### Enriching Crossmatch Results

After crossmatching, use `ParamFiller` with one or more parameter sources to fill stellar parameters, derive missing luminosity/flux/equilibrium-temperature values, estimate broad `msini` radius bounds, and apply rocky/temperate masks. The built-in `EMCCatalog.ENRICH_KEYS` and `NEACatalog.ENRICH_KEYS` mappings keep catalogue column names aligned with the enrichment API; see [docs/enrichment.md](docs/enrichment.md) for examples and caveats.


## Link to TAP services table definitions
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [SIMBAD ](https://simbad.u-strasbg.fr/simbad/tap/tapsearch.html)
- [Exo-MerCat](https://exo-mercat.readthedocs.io/en/latest/run_tap.html) (at time of writing, is not kept up to date, latest exomercat.csv file in directory) 
