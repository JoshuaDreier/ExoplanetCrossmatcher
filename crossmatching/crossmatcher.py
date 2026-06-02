import astropy
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord
import astropy.units as u

from crossmatching.catalogs.base import CatalogBase
from crossmatching.catalogs.nea import NEACatalog
from crossmatching.id_suppliers.base import IdSupplierBase
from crossmatching.id_suppliers.simbad import SimbadIdSupplier
from crossmatching.config import crossmatcher as _cm_cfg


def allowed_angular_separation(
        proper_motion: ArrayLike,
        pm_err: ArrayLike,
        epoch: ArrayLike,
        hpic_epoch: float = 2000,
        minimum: u.Quantity = 10*u.arcsec,
        unknown_default: u.Quantity = 50*u.arcsec,
    ) -> u.Quantity:
    """
    inputs must be in arcsec/yr. epoch and proper_motion may be masked arrays.
    Rows where epoch or proper_motion is masked (unknown) get unknown_default
    instead of the computed value.
    """
    epoch_arr = np.ma.asarray(epoch)
    pm_arr = np.ma.asarray(proper_motion)
    unknown = np.ma.getmaskarray(epoch_arr) | np.ma.getmaskarray(pm_arr)

    dt = np.abs(np.ma.filled(epoch_arr, hpic_epoch) - hpic_epoch)
    pm = np.ma.filled(pm_arr, 0.0)
    pmerr = np.ma.filled(np.ma.asarray(pm_err), 0.0)

    computed = (pm + pmerr) * dt * u.arcsec + minimum
    return np.where(
        unknown,
        unknown_default.to(u.arcsec).value,
        computed.to(u.arcsec).value
    ) * u.arcsec


class Crossmatcher:
    """
    Combines ID-based and coordinate-based crossmatching.

    First matches objects by ID, then performs coordinate-based
    matching on remaining unmatched sources.

    Parameters
    ----------
    catalog : CatalogBase, optional
        Catalog adapter providing schema and loading logic.
    id_supplier : IdSupplierBase, optional
        Alternate ID supplier. 
    """

    def __init__(
        self,
        catalog: CatalogBase,
        id_supplier: IdSupplierBase,
        coordinate_search_radius: u.Quantity = 10*u.arcsec,
        input_suffix: str = "input",
        **kwargs
    ):
        self.catalog = catalog
        self.id_supplier = id_supplier
        self.coordinate_search_radius = coordinate_search_radius
        self.input_suffix = input_suffix
        self.match_type_key   = _cm_cfg["match_type_key"] if "match_type_key" not in kwargs else kwargs["match_type_key"]
        self.id_match_label   = _cm_cfg["id_match_label"] if "id_match_label" not in kwargs else kwargs["id_match_label"]
        self.coord_match_label = _cm_cfg["coord_match_label"] if "coord_match_label" not in kwargs else kwargs["coord_match_label"]
        self.angular_sep_key  = _cm_cfg["angular_sep_key"] if "angular_sep_key" not in kwargs else kwargs["angular_sep_key"]
        self.catalog_table: Table = None
        self.alternate_ids: Table = None
        self._ids_for_names: frozenset | None = None
        self.id_matched: Table = None
        self.coords2d_matched: Table = None
        self.matched: Table = None
        self.planet_uuid = self.catalog.planet_uuid

    def _cache_catalog(self, table: Table) -> None:
        self.catalog_table = table

    def _cache_alternate_ids(self, table: Table, name_list: list[str]) -> None:
        self.alternate_ids = table
        self._ids_for_names = frozenset(name_list)

    def load_catalog(self, from_file=None, format="ascii", **kwargs) -> Table:
        self._cache_catalog(self.catalog.load(from_file=from_file, format=format, **kwargs))
        return self.catalog_table

    def load_alternate_ids(self, name_list, from_file=None) -> Table:
        self._cache_alternate_ids(
            self.id_supplier.load_alternate_ids(name_list, from_file=from_file),
            name_list,
        )
        return self.alternate_ids

    def id_crossmatch(self, input_table: Table, input_starname_key: str):
        name_list = input_table[input_starname_key].tolist()
        name_set = frozenset(name_list)
        if self.alternate_ids is None or not name_set <= self._ids_for_names:
            self.load_alternate_ids(name_list)
        elif name_set < self._ids_for_names:
            self._cache_alternate_ids(
                self.alternate_ids[np.isin(self.alternate_ids[self.id_supplier.input_col], name_list)],
                name_list,
            )
        if self.catalog_table is None:
            self.load_catalog()

        host_key = self.catalog.host_key

        # Collapse runs of whitespace before joining. Catalogs don't agree on internal spacing:
        # Example: SIMBAD stores "Ross  128" (two spaces) while NEA stores "Ross 128" (one space), so an
        # exact-string join silently drops the match without this normalization.
        input_col = self.id_supplier.input_col
        id_col = self.id_supplier.id_col

        cat_df = self.catalog_table.to_pandas()
        cat_df[host_key] = cat_df[host_key].str.split().str.join(" ")
        alt_df = self.alternate_ids.to_pandas()
        alt_df[id_col] = alt_df[id_col].str.split().str.join(" ")

        catalog_projected_onto_ids = cat_df.merge(
            alt_df,
            left_on=host_key,
            right_on=id_col,
            how="inner"
        )

        self.id_matched = input_table.to_pandas() 
        overlapping_columns = set(input_table.colnames) & set(self.catalog_table.colnames) - {input_starname_key}
        if overlapping_columns:
            self.id_matched.rename(columns={c: f"{c}_{self.input_suffix}" for c in overlapping_columns}, inplace=True)
        self.id_matched = self.id_matched.merge(
            catalog_projected_onto_ids,
            left_on=input_starname_key,
            right_on=input_col,
            how="inner"
        )

        self.id_matched.drop(columns=[input_col, id_col], inplace=True)
        self.id_matched[self.match_type_key] = self.id_match_label
        self.id_matched = Table.from_pandas(self.id_matched)

        # When the merge result is empty, Astropy can infer string columns as float64.
        # Cast only actual string columns back to strings so vstack can merge correctly.
        string_columns = {
            name
            for name in self.catalog_table.colnames
            if self.catalog_table[name].dtype.kind in {"U", "S", "O"}
        } | {
            name
            for name in self.alternate_ids.colnames
            if self.alternate_ids[name].dtype.kind in {"U", "S", "O"}
        } | {input_starname_key, self.match_type_key}
        for colname in self.id_matched.colnames:
            if colname in string_columns and self.id_matched[colname].dtype.kind in {"f", "i"}:
                self.id_matched[colname] = self.id_matched[colname].astype("U")
                self.id_matched[colname] = np.char.strip(self.id_matched[colname])

        self.id_matched[input_starname_key] = self.id_matched[input_starname_key].astype(str)
        return self.id_matched

    def find_duplicates(self, input_table: Table, input_starname_key: str, full=False) -> Table:
        name_list = input_table[input_starname_key].tolist()
        name_set = frozenset(name_list)
        if self.alternate_ids is None or not name_set <= self._ids_for_names:
            self.load_alternate_ids(name_list)
        elif name_set < self._ids_for_names:
            self._cache_alternate_ids(
                self.alternate_ids[np.isin(self.alternate_ids[self.id_supplier.input_col], name_list)],
                name_list,
            )

        input_col = self.id_supplier.input_col
        id_col_name = self.id_supplier.id_col
        id_col = self.alternate_ids[id_col_name]
        valid = (id_col != "") & (id_col != self.id_supplier.null_sentinel)
        alt_ids_without_nulls = self.alternate_ids[valid]

        linked_col = f"{input_col}_linked"
        self_joined = alt_ids_without_nulls.to_pandas().merge(
            alt_ids_without_nulls.to_pandas(),
            left_on=id_col_name,
            right_on=id_col_name,
            how="inner",
            suffixes=("", "_linked")
        )

        grouped: pd.DataFrame = self_joined.groupby(input_col)[linked_col].unique().reset_index()
        grouped.rename(columns={linked_col: "duplicate_names"}, inplace=True)
        grouped['duplicate_names'] = grouped['duplicate_names'].apply(lambda x: list(sorted(x)))
        grouped['appearances'] = grouped['duplicate_names'].apply(len)
        grouped = grouped[grouped['appearances'] > 1]
        grouped = grouped.sort_values(by='duplicate_names').reset_index(drop=True)
        if full:
            return Table.from_pandas(grouped)
        grouped = grouped.drop_duplicates(subset='duplicate_names')
        return Table.from_pandas(grouped)

    def remove_duplicates(self, input_table: Table, input_starname_key: str) -> Table:
        """Keep the row with the highest number of non-null values."""
        duplicates = self.find_duplicates(input_table, input_starname_key)
        drop_indices = []
        for dupe in duplicates:
            dupe_names = dupe["duplicate_names"]
            dupes_index_comparison_array = np.array([input_table[input_starname_key].data == name for name in dupe_names])

            for arr, name in zip(dupes_index_comparison_array, dupe_names):
                if np.sum(arr) != 1:
                    raise ValueError(f"Expected exactly one match for each duplicate name, but found {np.sum(arr)} matches for {name}")

            dupe_indeces = np.where(dupes_index_comparison_array)[1]
            null_counts = [
                np.isin(list(input_table[idx]), ["", 'null', '0', 0, None]).sum()
                for idx in dupe_indeces
            ]

            first_minimum_of_null = np.argmax(np.array(null_counts) == min(null_counts))
            drop_indices.extend(idx for i, idx in enumerate(dupe_indeces) if i != first_minimum_of_null)

        print(f"Removed Rows with indecies and names: {', '.join(str(i) for i in drop_indices)}")
        copy = input_table.copy()
        copy.remove_rows(drop_indices)
        return copy

    def coordinate_crossmatch(self, input_table: Table, input_starname_key: str, ra_key="ra", dec_key="dec") -> Table:
        if self.catalog_table is None:
            self.load_catalog()

        if 'coord_epoch' in self.catalog_table.colnames:
            epoch = self.catalog_table['coord_epoch']
        else:
            n = len(self.catalog_table)
            epoch = MaskedColumn(
                np.ma.MaskedArray(np.zeros(n), mask=np.ones(n, dtype=bool)),
                name='coord_epoch',
                description='Estimated epoch of sky coordinates (Julian year)',
            )

        pm_key = self.catalog.pm_key
        pmerr_key = self.catalog.pmerr_key
        if pm_key is not None and pmerr_key is not None:
            pm = self.catalog_table[pm_key] / 1000
            pmerr = self.catalog_table[pmerr_key] / 1000
        else:
            # No proper motion data — create all-masked arrays so unknown_default is used
            n = len(self.catalog_table)
            pm = np.ma.MaskedArray(np.zeros(n), mask=np.ones(n, dtype=bool))
            pmerr = np.ma.MaskedArray(np.zeros(n), mask=np.ones(n, dtype=bool))

        per_row_radius_2d = allowed_angular_separation(
            pm, pmerr,
            epoch,
            minimum=self.coordinate_search_radius
        )

        cat_ra_key = self.catalog.ra_key
        cat_dec_key = self.catalog.dec_key

        coords_input = SkyCoord(ra=input_table[ra_key]*u.deg, dec=input_table[dec_key]*u.deg)
        coords_catalog = SkyCoord(ra=self.catalog_table[cat_ra_key]*u.deg, dec=self.catalog_table[cat_dec_key]*u.deg)
        idx2d, sep2d, _ = coords_catalog.match_to_catalog_sky(coords_input)
        sep2d_mask = sep2d <= per_row_radius_2d

        input_matched_slice = input_table[idx2d[sep2d_mask]].copy()
        overlapping_columns = list(
            set(input_table.colnames) & set(self.catalog_table.colnames) - {input_starname_key}
        )
        if overlapping_columns:
            input_matched_slice.rename_columns(
                overlapping_columns,
                [f"{c}_{self.input_suffix}" for c in overlapping_columns]
            )

        self.coords2d_matched = astropy.table.hstack(
            [input_matched_slice, self.catalog_table[sep2d_mask]],
            join_type="exact"
        )
        self.coords2d_matched[self.match_type_key] = self.coord_match_label
        self.coords2d_matched[self.angular_sep_key] = sep2d[sep2d_mask]
        return self.coords2d_matched

    def combined_crossmatch(self, input_table: Table, input_starname_key: str):
        uuid = self.planet_uuid

        id_results = self.id_crossmatch(input_table, input_starname_key)
        coord_results = self.coordinate_crossmatch(input_table, input_starname_key)

        only_coords = coord_results[
            ~np.isin(coord_results[uuid].tolist(), id_results[uuid].tolist())
        ]

        self.matched = astropy.table.vstack([id_results, only_coords], join_type="outer")
        return self.matched


