import astropy
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord, angular_separation
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
        input_epoch: float = 2000,
        minimum: u.Quantity = 10*u.arcsec,
        unknown_default: u.Quantity = 50*u.arcsec,
    ) -> u.Quantity:
    """Compute per-row coordinate search radii from proper-motion data.

    The radius for row *i* grows with the proper-motion displacement
    accumulated between the catalog coordinate epoch and the input survey
    epoch::

        radius_i = (pm_i + pm_err_i) * |epoch_i - hpic_epoch| + minimum

    Rows where ``epoch`` or ``proper_motion`` is masked (unknown) receive
    ``unknown_default`` instead of the computed value.

    Parameters
    ----------
    proper_motion : array-like
        Total proper motion in arcsec/yr.  May be a masked array;
        masked rows are treated as unknown and receive
        ``unknown_default``.
    pm_err : array-like
        Proper-motion uncertainty in arcsec/yr.  May be a masked array;
        masked rows contribute 0 to the uncertainty term.
    epoch : array-like
        Coordinate epoch of the catalog positions (Julian year).  May
        be a masked array; masked rows trigger ``unknown_default``.
    input_epoch : float, optional
        Epoch of the input survey positions (Julian year).  Default
        2000 (HPIC LC4 reference epoch).
    minimum : `~astropy.units.Quantity`, optional
        Minimum search radius added to every computed value.
        Default 10 arcsec.
    unknown_default : `~astropy.units.Quantity`, optional
        Search radius assigned to rows with unknown proper motion or
        epoch.  Default 50 arcsec.

    Returns
    -------
    radii : `~astropy.units.Quantity`
        Per-row search radii in arcsec, shape matching the input arrays.
    """
    epoch_arr = np.ma.asarray(epoch)
    pm_arr = np.ma.asarray(proper_motion)
    pm_err_arr = np.ma.asarray(pm_err)
    unknown = np.ma.getmaskarray(epoch_arr) | np.ma.getmaskarray(pm_arr) | np.ma.getmaskarray(pm_err_arr)

    dt = np.abs(np.ma.filled(epoch_arr, input_epoch) - input_epoch)
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
    catalog : CatalogBase
        Catalog adapter providing schema and loading logic.
        Availabe Catalogs: EMCCatalog (Exo-MerCat), NEACatalog (Nasa Exomplanet Archive), 
        Custom Catalogs can be implemented by loading a file using FileCatalog or by
        extending the CatalogBase class.
    id_supplier : IdSupplierBase
        supplier of the alisases used for ID crossmatching. 
        Available ID suppliers: SimbadIDSupplier, EMCID
        Custom ID suppliers can be implemented by extending the IdSupplierBase class.
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

    def _ensure_alternate_ids(self, name_list: list[str]) -> None:
        """Load or subset the alternate-ID cache to cover exactly name_list."""
        name_set = frozenset(name_list)
        if self.alternate_ids is None or not name_set <= self._ids_for_names:
            self.load_alternate_ids(name_list)
        elif name_set < self._ids_for_names:
            self._cache_alternate_ids(
                self.alternate_ids[np.isin(self.alternate_ids[self.id_supplier.input_col], name_list)],
                name_list,
            )

    def load_catalog(self, from_file: str | None = None, format: str = "ascii", **kwargs) -> Table:
        """Download or read the planet catalog and cache it.

        Parameters
        ----------
        from_file : str, optional
            Path to a previously saved raw catalog file.  If given,
            reads from disk; otherwise calls the catalog's
            :meth:`~CatalogBase.download`.
        format : str, optional
            Astropy table format string (default ``'ascii'``).
        **kwargs
            Forwarded to :meth:`~CatalogBase.load`.

        Returns
        -------
        table : `~astropy.table.Table`
            Preprocessed catalog table, also stored as
            ``self.catalog_table``.
        """
        self._cache_catalog(self.catalog.load(from_file=from_file, format=format, **kwargs))
        return self.catalog_table

    def load_alternate_ids(self, name_list: list[str], from_file: str | None = None) -> Table:
        """Fetch or read alternate star identifiers and cache them.

        Parameters
        ----------
        name_list : list of str
            Input star names to load IDs for.
        from_file : str, optional
            Path to a previously saved alternate-ID file.  If given,
            reads from disk; otherwise queries the ID supplier's remote
            source.

        Returns
        -------
        ids : `~astropy.table.Table`
            Two-column table (``input_col``, ``id_col``), also stored
            as ``self.alternate_ids``.
        """
        self._cache_alternate_ids(
            self.id_supplier.load_alternate_ids(name_list, from_file=from_file),
            name_list,
        )
        return self.alternate_ids

    def id_crossmatch(
            self, 
            input_table: Table, 
            input_starname_key: str,
            ra_key: str | None = None,
            dec_key: str | None = None,
            ra_unit: u.Unit = u.deg,
            dec_unit: u.Unit = u.deg,
        ):
        """Match input stars to the planet catalog by alternate identifier.

        Fetches alternate IDs for all input stars (or uses the cached
        set), then inner-joins the catalog on normalised host-star names.
        Whitespace in all name columns is collapsed to single spaces
        before joining, so that e.g. ``'Ross  128'`` (SIMBAD) matches
        ``'Ross 128'`` (NEA).

        Results are stored in ``self.id_matched`` and also returned.

        Parameters
        ----------
        input_table : `~astropy.table.Table`
            Stellar input table.  Must contain a column named
            ``input_starname_key``.
        input_starname_key : str
            Name of the column in ``input_table`` that holds star names.
        ra_key : str or None, optional
            Column name for right ascension in ``input_table``.  If
            given (together with ``dec_key``), the on-sky angular
            separation between each matched pair is computed and stored
            in ``angular_sep_key``.  Default ``None`` (no separation
            computed).
        dec_key : str or None, optional
            Column name for declination in ``input_table``.  See
            ``ra_key``.  Default ``None``.
        ra_unit : `~astropy.units.Unit`, optional
            Unit of the ``ra_key`` column.  Default degrees.
        dec_unit : `~astropy.units.Unit`, optional
            Unit of the ``dec_key`` column.  Default degrees.

        Returns
        -------
        matched : `~astropy.table.Table`
            Inner-joined table of input rows × catalog planet rows.
            Added columns:

            - ``match_type`` : ``'id'`` for every row.
            - ``angular_separation`` : `~astropy.units.Quantity` (arcsec),
              great-circle distance between the matched input and catalog
              positions.  Present only when ``ra_key`` and ``dec_key``
              are supplied.

            Columns present in both input and catalog (except the name
            key) are suffixed with ``'_input'`` on the input side.
        """
        name_list = input_table[input_starname_key].tolist()
        self._ensure_alternate_ids(name_list)
        if self.catalog_table is None:
            self.load_catalog()

        host_key = self.catalog.host_key

        # Collapse runs of whitespace before joining. Catalogs don't agree on internal spacing:
        # Example: SIMBAD stores "Ross  128" (two spaces) while NEA stores "Ross 128" (one space), so an
        # exact-string join silently drops the match without this normalization.
        input_col = self.id_supplier.input_col
        id_col = self.id_supplier.id_col

        cat_df = self.catalog_table.to_pandas()
        cat_df[host_key]  = cat_df[host_key].str.split().str.join(" ")
        alt_df = self.alternate_ids.to_pandas()
        alt_df[id_col]    = alt_df[id_col].str.split().str.join(" ")
        alt_df[input_col] = alt_df[input_col].str.split().str.join(" ")

        catalog_projected_onto_ids = cat_df.merge(
            alt_df,
            left_on=host_key,
            right_on=id_col,
            how="inner"
        )

        self.id_matched = input_table.to_pandas()
        self.id_matched[input_starname_key] = self.id_matched[input_starname_key].str.split().str.join(" ")
        overlapping_columns = (set(input_table.colnames) & set(self.catalog_table.colnames)) - {input_starname_key}
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
        
        if ra_key is not None and dec_key is not None:
            # id_crossmatch renames any input column that collides with a catalog column
            # (e.g. "ra" → "ra_input").  Resolve the actual names before indexing.
            input_ra_col  = f"{ra_key}_{self.input_suffix}"  if ra_key  in self.catalog_table.colnames else ra_key
            input_dec_col = f"{dec_key}_{self.input_suffix}" if dec_key in self.catalog_table.colnames else dec_key

            self.id_matched[self.angular_sep_key] = angular_separation(
                self.id_matched[input_ra_col]*ra_unit,       
                self.id_matched[input_dec_col]*dec_unit,      
                self.id_matched[self.catalog.ra_key]*self.catalog.ra_unit,
                self.id_matched[self.catalog.dec_key]*self.catalog.dec_unit,
            ).to(u.arcsec)
        
        return self.id_matched

    def find_duplicates(self, input_table: Table, input_starname_key: str, full: bool = False) -> Table:
        """Find input stars that share alternate identifiers.

        Performs a self-join on the alternate-ID table to identify groups
        of input star names that share at least one common identifier
        (i.e. refer to the same astronomical object).

        Parameters
        ----------
        input_table : `~astropy.table.Table`
            Stellar input table.  Must contain a column named
            ``input_starname_key``.
        input_starname_key : str
            Name of the column in ``input_table`` that holds star names.
        full : bool, optional
            If ``False`` (default), return one representative row per
            duplicate group.  If ``True``, return one row per member of
            each group (groups with *n* members contribute *n* rows).

        Returns
        -------
        duplicates : `~astropy.table.Table`
            Table with columns ``input_col`` (star name),
            ``duplicate_names`` (sorted list of all names sharing an ID),
            and ``appearances`` (group size).  Empty if no duplicates
            are found.
        """
        name_list = input_table[input_starname_key].tolist()
        self._ensure_alternate_ids(name_list)

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
        """Remove duplicate input rows, keeping the most data-complete copy.

        For each group of stars sharing an alternate identifier, retains
        the row with the fewest null or sentinel values (``''``,
        ``'null'``, ``'0'``, ``0``, ``None``).  Ties are broken by
        keeping the first minimum.

        Parameters
        ----------
        input_table : `~astropy.table.Table`
            Stellar input table containing potential duplicate rows.
        input_starname_key : str
            Name of the column in ``input_table`` that holds star names.

        Returns
        -------
        deduplicated : `~astropy.table.Table`
            Copy of ``input_table`` with one row removed per duplicate
            group (keeping the most complete entry).

        Raises
        ------
        ValueError
            If a name in a duplicate group does not appear exactly once
            in ``input_table``.

        Notes
        -----
        Prints the indices and names of removed rows to stdout.
        """
        duplicates = self.find_duplicates(input_table, input_starname_key)
        drop_indices = []
        for dupe in duplicates:
            dupe_names = dupe["duplicate_names"]
            dupes_index_comparison_array = np.array([input_table[input_starname_key].data == name for name in dupe_names])

            for arr, name in zip(dupes_index_comparison_array, dupe_names):
                if np.sum(arr) != 1:
                    raise ValueError(f"Expected exactly one match for each duplicate name, but found {np.sum(arr)} matches for {name}")

            dupe_indices = np.where(dupes_index_comparison_array)[1]
            null_counts = [
                np.isin(list(input_table[idx]), ["", 'null', '0', 0, None]).sum()
                for idx in dupe_indices
            ]

            first_minimum_of_null = np.argmax(np.array(null_counts) == min(null_counts))
            drop_indices.extend(idx for i, idx in enumerate(dupe_indices) if i != first_minimum_of_null)

        print(f"Removed Rows with indices and names: {', '.join(str(i) for i in drop_indices)}")
        copy = input_table.copy()
        copy.remove_rows(drop_indices)
        return copy

    def coordinate_crossmatch(
            self,
            input_table: Table,
            input_starname_key: str, 
            ra_key: str = "ra", 
            ra_unit: u.Quantity = u.degree,
            dec_key: str = "dec",
            dec_unit: u.Quantity = u.degree
        ) -> Table:
        """Match input stars to the planet catalog by 2D sky coordinates.

        For each catalog planet host, finds the nearest input star within
        a per-row angular search radius.  The radius grows with the
        proper-motion displacement accumulated since the catalog's
        coordinate epoch::

            radius_i = (pm_i + pm_err_i) * |epoch_i - 2000| + coordinate_search_radius

        Catalog proper-motion values (``pm_key``, ``pmerr_key``) are
        converted to arcsec/yr using ``catalog.pm_unit``.  Rows with unknown
        proper motion or epoch receive ``unknown_default`` = 50 arcsec.

        Results are stored in ``self.coords2d_matched`` and also returned.

        Parameters
        ----------
        input_table : `~astropy.table.Table`
            Stellar input table.  Must contain ``ra_key`` and ``dec_key``
            columns in degrees.
        input_starname_key : str
            Name of the column in ``input_table`` that holds star names.
            Used only to detect and suffix overlapping column names.
        ra_key : str, optional
            Column name for right ascension in ``input_table`` (in units of ``ra_unit``).
            Default ``'ra'``.
        ra_unit : `u.Unit` 
            Unit of ``ra_key`` column in ``input_table``
            Defaults to degrees.
        dec_key : str, optional
            Column name for declination in ``input_table``.
            Default ``'dec'``.
        dec_unit : `u.Unit`
            Unit of ``dec_key`` column in ``input_table``.
            Defaults to degrees.

        Returns
        -------
        matched : `~astropy.table.Table`
            Horizontally stacked table of input rows × catalog planet
            rows for all matches within the per-row radius.  Added
            columns:

            - ``match_type`` : ``'coordinates'`` for every row
            - ``angular_separation`` : `~astropy.units.Quantity` (arcsec),
              angular distance between the matched input and catalog
              positions
        """
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
            pm = np.ma.asarray(self.catalog_table[pm_key])*self.catalog.pm_unit.to(u.arcsec / u.yr)
            pmerr = np.ma.asarray(self.catalog_table[pmerr_key])*self.catalog.pm_unit.to(u.arcsec / u.yr)
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

        coords_input = SkyCoord(
            ra=input_table[ra_key]*ra_unit,
            dec=input_table[dec_key]*dec_unit
        )
        coords_catalog = SkyCoord(
            ra=self.catalog_table[cat_ra_key]*self.catalog.ra_unit,
            dec=self.catalog_table[cat_dec_key]*self.catalog.dec_unit
        )
        idx2d, sep2d, _ = coords_catalog.match_to_catalog_sky(coords_input)
        sep2d = sep2d.to(u.arcsec)
        sep2d_mask = (sep2d <= per_row_radius_2d)

        input_matched_slice = input_table[idx2d[sep2d_mask]].copy()
        overlapping_columns = list(
            (set(input_table.colnames) & set(self.catalog_table.colnames)) - {input_starname_key}
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

    def combined_crossmatch(
            self,
            input_table: Table,
            input_starname_key: str,
            ra_key: str = "ra",
            ra_unit: u.Unit = u.degree,
            dec_key: str = "dec",
            dec_unit: u.Unit = u.degree,
        ):
        """Run ID and coordinate matching and merge the results.

        Executes :meth:`id_crossmatch` first, then
        :meth:`coordinate_crossmatch`.  Planets already found by ID
        matching are removed from the coordinate results (deduplication
        by ``planet_uuid``) before the two tables are stacked.  This
        ensures each planet row appears at most once, with ID matches
        taking priority.

        Results are stored in ``self.matched`` and also returned.

        Parameters
        ----------
        input_table : `~astropy.table.Table`
            Stellar input table.  Must contain ``ra_key`` and ``dec_key``
            columns plus a column named ``input_starname_key``.
        input_starname_key : str
            Name of the column in ``input_table`` that holds star names.
        ra_key : str, optional
            Column name for right ascension in ``input_table``.
            Default ``'ra'``.
        ra_unit : `~astropy.units.Unit`, optional
            Unit of the ``ra_key`` column.  Default degrees.
        dec_key : str, optional
            Column name for declination in ``input_table``.
            Default ``'dec'``.
        dec_unit : `~astropy.units.Unit`, optional
            Unit of the ``dec_key`` column.  Default degrees.

        Returns
        -------
        matched : `~astropy.table.Table`
            Combined table of all matched planet rows.  Added columns:

            - ``match_type`` : ``'id'`` or ``'coordinates'``
            - ``angular_separation`` : `~astropy.units.Quantity` (arcsec),
              great-circle distance between the matched input and catalog
              positions (present for both ID and coordinate matches).
        """
        uuid = self.planet_uuid

        id_results = self.id_crossmatch(
            input_table,
            input_starname_key,
            ra_key=ra_key,
            ra_unit=ra_unit,
            dec_key=dec_key,
            dec_unit=dec_unit
        )
        coord_results = self.coordinate_crossmatch(
            input_table, input_starname_key,
            ra_key=ra_key, ra_unit=ra_unit, dec_key=dec_key, dec_unit=dec_unit,
        )

        only_coords = coord_results[
            ~np.isin(coord_results[uuid], id_results[uuid])
        ]

        self.matched = astropy.table.vstack([id_results, only_coords], join_type="outer")
        return self.matched


