# Catalogs

A catalog adapter wraps a planet data source and provides a uniform interface to `Crossmatcher`.  All adapters extend `CatalogBase` and expose the same `load(from_file=...)` pattern.

## NEACatalog
Downloads the NASA Exoplanet Archive `pscomppars` composite-parameters table.

```python
from crossmatching import NEACatalog

catalog = NEACatalog()
table = catalog.load(from_file="pscomppars.txt")  # from cache
# or: table = catalog.load()                      # downloads from TAP (~1.5 min)
```

### Schema keys

| Attribute     | Value         | Column description                 |
| ------------- | ------------- | ---------------------------------- |
| `ra_key`      | `'ra'`        | Right ascension (degrees, J2000)   |
| `dec_key`     | `'dec'`       | Declination (degrees, J2000)       |
| `host_key`    | `'hostname'`  | Host-star name (join key)          |
| `planet_uid` | `'pl_name'`   | Unique planet name                 |
| `pm_key`      | `'sy_pm'`     | Total proper motion (mas/yr)       |
| `pmerr_key`   | `'sy_pmerr1'` | Proper-motion uncertainty (mas/yr) |

**`coord_epoch` column added by `preprocess()`**

`NEACatalog.preprocess()` adds a `coord_epoch` column estimated for each row
using this priority order:

1. Gaia DR3 or DR2 cross-ID present → **2016.0**
2. `ra_reflink` mentions TICv8 / Stassun → **2000.0**
3. `ra_reflink` mentions Hipparcos → **1991.25**
4. `ra_reflink` publication year ≥ 2018 → **2016.0**
5. `ra_reflink` publication year < 2018 → **that year**
6. None of the above → **masked** (coordinate matching uses `default_search_radius` fallback)

**Saving a local cache**

```python
NEACatalog().save_raw("pscomppars.txt")
```


## EMCCatalog

Reads the Exo-MerCat merged catalog from a local CSV file. 
Note that at time of writing the [Exo-MerCat TAP](https://exo-mercat.readthedocs.io/en/latest/run_tap.html) service is not kept up to date (and missing the alias column), so ideally clone [Exo-MerCat](https://github.com/Exo-MerCat/Exo-MerCat), and copy the resulting `exomarcat.csv`file to this directory. A timestamped version is provided in this projects `input/` directory.

```python
from crossmatching import EMCCatalog

catalog = EMCCatalog()
table = catalog.load(from_file="exo-mercat.csv")
```


## FileCatalog

Wraps any local planet table with a user-specified column mapping.  Useful for
custom catalogs or pre-processed files from other pipelines.

```python
from crossmatching import FileCatalog

catalog = FileCatalog(
    path="my_planets.csv",
    ra_key="ra_deg",
    dec_key="dec_deg",
    host_key="host_star",
    planet_uid="planet_id",
    pm_key=None,       # no proper motion in this file
    pmerr_key=None,
    format="ascii.csv",
)
table = catalog.load()
```

`from_file=` overrides the constructor path for a one-off load.

## Implementing a custom catalog

Subclass `CatalogBase`, declare the schema attributes, and implement `download()`.

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

Override `preprocess()` if you need to add derived columns (e.g. `coord_epoch`).

---

## Caching pattern

All three built-in catalogs follow the same pattern.

| Method                               | Network? | Use case             |
| ------------------------------------ | -------- | -------------------- |
| `catalog.load()`                     | yes      | First download       |
| `catalog.load(from_file="file.txt")` | no       | Subsequent runs      |
| `catalog.save_raw("file.txt")`       | yes      | Download and persist |
