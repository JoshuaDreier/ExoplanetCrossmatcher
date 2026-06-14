# Configuration

`crossmatching/crossmatching.cfg` provides project-wide defaults for column
names and match-type labels.  The file is read at import time; values can be
overridden per-instance via keyword arguments to `Crossmatcher`.

## Default values

### `[id_supplier]` section

| Key             | Default     | Meaning                                                                    |
| --------------- | ----------- | -------------------------------------------------------------------------- |
| `input_col`     | `input_ids` | Column name for input star names in the alternate-ID table                 |
| `id_col`        | `id`        | Column name for alternate identifiers                                      |
| `null_sentinel` | `--`        | Placeholder string meaning "no identifier available", default from astropy |

### `[crossmatcher]` section

| Key | Default | Meaning |
|-----|---------|---------|
| `match_type_key` | `match_type` | Output column distinguishing the match strategy |
| `id_match_label` | `id` | Value written to `match_type` for ID-based matches |
| `coord_match_label` | `coordinates` | Value written to `match_type` for coordinate matches |
| `angular_sep_key` | `angular_separation` | Output column for angular distance (arcsec, coordinate matches only) |

## Overriding per instance

Pass keyword arguments to `Crossmatcher` to override any config value for that
instance without editing the file:

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

## Editing `crossmatching.cfg`

To change defaults project-wide, edit `crossmatching/crossmatching.cfg`.  The
file is a standard Python `configparser` INI file:

```ini
[id_supplier]
input_col = input_ids
id_col = id
null_sentinel = --

[crossmatcher]
match_type_key = match_type
id_match_label = id
coord_match_label = coordinates
angular_sep_key = angular_separation
```
