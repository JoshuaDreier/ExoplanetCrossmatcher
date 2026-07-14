# Configuration

The schema-key defaults (output column names and match-type labels) are plain
defaults baked into the code — there is no config file.  Override them
per-instance with keyword arguments; the defaults are the de-facto standard
used throughout the project.

## `Crossmatcher` schema keys

These are constructor keyword arguments of `Crossmatcher`:

| Keyword | Default | Meaning |
|---------|---------|---------|
| `match_type_key` | `match_type` | Output column distinguishing the match strategy |
| `id_match_label` | `id` | Value written to `match_type` for ID-based matches |
| `coord_match_label` | `coordinates` | Value written to `match_type` for coordinate matches |
| `angular_sep_key` | `crossmatching_angular_separation` | Output column for angular distance (arcsec) |

Rows found by both strategies are labelled `f"{id_match_label}+{coord_match_label}"`
(default `id+coordinates`).

```python
cm = Crossmatcher(
    catalog=NEACatalog(),
    id_supplier=SimbadIdSupplier(),
    match_type_key="method",         # rename the match-type column
    id_match_label="identifier",     # rename the label for ID matches
    coord_match_label="sky_position",
    angular_sep_key="sep_arcsec",
)
```

## ID-supplier schema keys

These are class attributes on `IdSupplierBase` (shared by `SimbadIdSupplier`
and `EMCIdSupplier`):

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `input_col` | `input_ids` | Column name for input star names in the alternate-ID table |
| `id_col` | `id` | Column name for alternate identifiers |
| `null_sentinel` | `--` | Placeholder meaning "no identifier available" (astropy's masked default) |

Override them by setting the attribute on a subclass or instance:

```python
supplier = SimbadIdSupplier()
supplier.input_col = "star_name"   # instance override

class MyIdSupplier(IdSupplierBase): # subclass override
    id_col = "identifier"
```
