# Detecting and removing duplicate input stars

Some input_table entries may refer to the same object under different names (e.g. `'Proxima Cen'` and `'alpha Cen C'`). Working only with ID aliases from the given `IdSupplier`, there are two methods to find and delete duplicates in an input table (no coordinate information is used):
```python
from crossmatching import EMCCatalog, EMCIdSupplier, Crossmatcher

cme = Crossmatcher(EMCCatalog(), EMCIdSupplier())

# Find groups of names that share an alternate identifier
dupes = cme.find_duplicates(hpic, input_starname_key="star_name")
print(dupes["star_name", "duplicate_names", "appearances"])

# Remove duplicates, keeping the most data-complete row per group
input_clean = cme.remove_duplicates(hpic, input_starname_key="star_name")
```

- `find_duplicates(full=True)` returns one row per group member instead of one representative row per group.
- `remove_duplicates` selects the row with the least `null/nan/masked` values and drops the rest


