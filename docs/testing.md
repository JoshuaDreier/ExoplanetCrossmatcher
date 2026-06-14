# Testing

## Running the tests

```bash
# All tests
pytest

# Single file
pytest tests/e2e/test_proxima_cen_b.py

# Tests matching a keyword
pytest -k "combined"

# Verbose output
pytest -v
```

`pytest.ini` sets `pythonpath = .` so imports resolve from the project root
without installation.

---

## Test pyramid

```
tests/
  unit/           # Fast, no network — tests individual functions in isolation
  functional/     # Medium — tests a component with real data but mocked I/O
  e2e/            # Slow, requires cached data files — full crossmatch pipeline
    guinea_tics/  # Gold-standard e2e tests for known exoplanet host stars
```

### Unit tests

Test individual pure functions: `allowed_angular_separation`, duplicate
detection, cache invalidation logic, column-schema validation.  These run in
milliseconds and have no external dependencies.

### Functional tests

Test a component (e.g. `StellarParamMerger`) with real table data loaded from
`tests/data/`.  Faster than e2e because they don't run the full crossmatch
pipeline.

### End-to-end tests

Run the actual crossmatch against the full cached catalog and verify that
specific known planets are found.  Require the versioned data snapshots in
`tests/data/` to be present on disk — currently `pscomppars_20260611.txt`,
`alternate_ids_hpic_20260611.txt`, and `HPIC_LC4_combined_d50_20260611.txt`.
The date suffix records when the snapshot was taken; update the filenames in
`tests/e2e/e2e_crossmatch_methods.py` and `tests/e2e/test_emc_superset.py`
whenever the snapshots are refreshed.

---

## The guinea-tics gold standard

`tests/data/guinea_tics.csv` is a hand-curated dataset of known exoplanet host
stars used as the ground truth for e2e tests.  Each row specifies:

| Column | Description |
|--------|-------------|
| `star_name` | HPIC input name |
| `expected_planets` | Pipe-delimited list of expected planet identifiers |
| `ra`, `dec` | Coordinates (degrees) |
| `sy_dist` | Distance (pc) |

### Adding a star to the gold standard

1. Add a row to `tests/data/guinea_tics.csv`.
2. Run `pytest -k "your_star_name"` — the test is auto-generated from the CSV.

### Assertion helpers

Two helpers in `tests/e2e/e2e_utils.py` make gold-standard assertions
non-fatal across multiple stars per test run:

- `_e2e_all_planets_found()` — checks that every expected planet appears in the
  crossmatch result
- `_e2e_no_false_positives()` — checks that every matched planet was expected

Both use `pytest-check` so all failures are collected before the test ends,
rather than stopping at the first missing planet.

### Fixtures

- `loaded_matcher` — session-scoped; loads the catalog and IDs once per test
  session to avoid redundant I/O
- `stateless_matcher` — resets `Crossmatcher` state between tests; use when
  a test modifies internal state

---

## What the e2e tests guarantee

- Every star in `guinea_tics.csv` produces at least the expected planet matches
  (no false negatives)
- No unexpected planets appear for those stars (no false positives)
- Both `id_crossmatch`, `coordinate_crossmatch`, and `combined_crossmatch` are
  covered for each entry

The tests do **not** guarantee performance on stars outside the gold standard,
nor do they test enrichment output.
