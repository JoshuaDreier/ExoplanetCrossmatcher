
# Crossmatching
`Crossmatcher` implements two complementary matching strategies that can be used independently or combined.  See [Architecture](architecture.md) for the overall data flow.

## Constructor

```python
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
import astropy.units as u

cm = Crossmatcher(
    catalog=NEACatalog(), # or EMCCatalog or self-extended Catalog
    id_supplier=SimbadIdSupplier(), # or EMCIdSupplier 
)
```

| Parameter               | Default   | Meaning                                                                 |
| ----------------------- | --------- | ----------------------------------------------------------------------- |
| `catalog`               | —         | Catalog adapter (see [Catalogs](catalogs.md))                           |
| `id_supplier`           | —         | Alternate-ID supplier (see [ID Suppliers](id-suppliers.md))             |
| `minimum_search_radius` | 10 arcsec | Minimum angular search radius; actual radius grows with proper motion   |
| `default_search_radius` | 50 arcsec | Default search                                                          |
| `input_suffix`          | `'input'` | Suffix appended to input columns that share a name with catalog columns |

## ID-based matching
`id_crossmatch()` finds planets whose catalog host-star name appears list of aliases of an input star.
The aliases are provided by the chosen `IdSupplier`.

### How it works
1. Fetch (or load from cache) all alternate IDs for the names of input stars (retrieved from `input_table[input_starname_key]`)
2. Collapse runs of whitespace in every name column to a single space.
   This is necessary because for exampele SIMBAD stores `'Ross  128'` (two spaces) while
   NEA stores `'Ross 128'` (one space) , which exact-string join would silently
   miss this match without normalisation.
3. Inner-join: `input_table` joined with  ` alternate_ids` joined with `catalog` on the normalised names.
4. Tag every result row with `match_type = 'id'`.

```python
id_result = cm.id_crossmatch(input_table, input_starname_key="star_name")
```

Each `IdSupplier` may implement rules for adding additional aliases, see [id-suppliers](id-suppliers.md) for details.

## Coordinate-based matching
`coordinate_crossmatch()` finds planets whose catalog host-star position falls within an angular radius of an input star's position.

The matching direction is: for each **catalog** row find the nearest **input** star. This means a single input star can match multiple planet rows if they share the same host.


### Per-row search radius
If the proper motion of the stars in the catalog are specified, the search radius is not fixed, it grows with the proper-motion displacement accumulated between the catalog's coordinate epoch and the input survey epoch.

```
radius_i = (pm_i + pm_err_i) × |epoch_i − input_epoch| + minimum_search_radius
```

The column keys and units of the proper motion values are taken from the `Catalog` attributes.

Rows where the epoch or proper motion is unknown receive a fallback radius of  `default_search_radius`, which is set to 50 arcsec by default.

### Usage
```python
coord_result = cm.coordinate_crossmatch(
    input_table,
    input_starname_key="star_name",
    input_epoch= 2000 # epoch of input_tables coordinate columns
    ra_key="ra",   # column name in input_table for Right Ascension (degrees)
	ra_unit = u.degree, # unit of Right Ascension in input_table
    dec_key ="dec", # column name in input_table for Declination
	dec_unit = u.degree #  unit of Declination in input_table
)
```


## Combined matching
`combined_crossmatch()` runs both strategies and deduplicates the results:

1. Run `id_crossmatch()`.
2. Run `coordinate_crossmatch()`.
3. Remove from the coordinate results any planet already found by ID (matched
   on `planet_uid`, e.g. `pl_name`).
4. Stack ID results on top of the filtered coordinate results.

```python
result = cm.combined_crossmatch(hpic, input_starname_key="star_name")
```

If a planet is found by both methods, it appears only once in the combined results, tagged with `match_type = 'id+coordinates'`.
[detect-duplicates](detect-duplicates.md)


## Output columns
All catalog columns pass through unchanged.  Added columns:

| Column                             | Type              | Note                                                                                                         |
| ---------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `match_type`                       | `str`             | Column name configurable via the `match_type_key` constructor kwarg                                          |
| `crossmatching_angular_separation` | Quantity (arcsec) | Only computed for `id_crossmatch()` if `ra_key, dec_key`are passed.<br>Column name configurable via the `angular_sep_key` constructor kwarg |

Columns that exist in both the input table and the catalog (other than the starname key) are suffixed with `_input` on the input side. (or whatever is passed to `input_suffix`)

See [Column Reference](column-reference.md) for the full schema.