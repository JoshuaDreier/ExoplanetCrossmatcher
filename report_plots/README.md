# report_plots

One notebook per paper output, named after the file it produces. Running all
cells of a notebook top-to-bottom rebuilds its output and exports it to
`report/figures/<notebook-name>.pdf` (or `.tex` for the tables).

Shared code:

- `common.py` repo root/`FIGURES_DIR` paths, `save_figure()`, HZ flux limits, spectral-category colors/order
- `data.py` input table, NEA/EMC crossmatchers, `ParamFiller` construction (NEA,EU,EPIC,TOI,SIMBAD Order), `enrich()`, `compute_is_rocky_temperate()`
- `plots.py` the plotting functions (`plot_distr`, `double_hist_heatmap`, `spectral_grid_heatmap`, `plot_rocky_temp_bars`)
- `tables.py`: `build_rocky_temperate_table_tex()` (tabularray + siunitx)

The catalog files in `input/` were sourced on 2026-07-11. With a local install of
Exo-MerCat, fresh snapshots can be generated and copied over the ones in `input/`
to update them.