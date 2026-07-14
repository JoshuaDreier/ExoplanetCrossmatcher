# Exoplanet Crossmatcher

This is a Python library that identifies exoplanet host stars by matching an input stellar
survey (with names and optionally coordinates) against exoplanet catalogs.  Two
complementary strategies are used: 
- identifier-based matching
- sky-coordinate matching with proper-motion-aware search radii.  

The results can
be enriched with stellar and derived planetary parameters from multiple sources.

## Navigation

| Page | Contents |
|------|----------|
| [Architecture](architecture.md) | Package layout, crossmatch data flow, enrichment pipeline diagram |
| [Crossmatching](crossmatching.md) | ID matching, coordinate matching, combined strategy, output columns |
| [Enrichment](enrichment.md) | `ParamFiller`, parameter-source priority chain, derived columns |
| [Derived Parameter Inference](derived-parameter-inference.md) | Formulas, provenance strings, and citations for enrichment fallbacks |
| [Catalogs](catalogs.md) | `NEACatalog`, `FileCatalog`, `EMCCatalog` — when to use each |
| [ID Suppliers](id-suppliers.md) | `SimbadIdSupplier`, `EMCIdSupplier` — how alternate IDs are fetched |
| [Configuration](configuration.md) | Schema-key defaults and constructor override kwargs |
| [Column Reference](column-reference.md) | Full schema for crossmatch and enrichment output tables |
| [Testing](testing.md) | Test pyramid, guinea-tics gold standard, how to run |

## External resources

- [NASA Exoplanet Archive TAP](https://exoplanetarchive.ipac.caltech.edu/TAP) — `pscomppars` table
- [SIMBAD TAP](https://simbad.cds.unistra.fr/simbad/sim-tap) — alternate identifiers
- [Exo-MerCat](https://github.com/Exo-MerCat/Exo-MerCat) — alternative merged catalog
