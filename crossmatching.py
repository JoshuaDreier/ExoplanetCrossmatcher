import astropy
import numpy as np
import pyvo
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Column, MaskedColumn
import re


def allowed_angular_seperation(proper_motion, pm_err, epoch, hpic_epoch=2000,
                               minimum=10*u.arcsec, unknown_default=50*u.arcsec):
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


def extract_year_from_reflink(reflink):
    """Parse a publication year from an ADS-style HTML reflink string."""
    if not reflink or np.ma.is_masked(reflink):
        return None

    # Try the structured refstr attribute first: e.g. STASSUN_ET_AL__2019
    match = re.search(r'refstr=["\']?[A-Z_]+_(\d{4})', reflink)
    if match:
        return int(match.group(1))

    # Fallback: any 4-digit year anywhere in the string
    match = re.search(r'\b(19|20)\d{2}\b', reflink)
    if match:
        return int(match.group(0))

    return None


def coord_epoch(reflink, gaia_dr3, gaia_dr2):
    """
    Estimate the coordinate epoch for a single catalogue row.
    Returns a float (Julian year) or None if the epoch cannot be determined.

    Priority:
      1. Gaia DR3 id present  -> 2016.0
      2. Gaia DR2 id present  -> 2016.0
      3. ra_reflink mentions TICv8/Stassun -> 2000.0
      4. ra_reflink mentions Hipparcos     -> 1991.25
      5. ra_reflink publication year >= 2018 (post-Gaia DR2) -> 2016.0
      6. ra_reflink publication year < 2018  -> that year (rough proxy)
      7. Default -> None (unknown; allowed_angular_seperation will use unknown_default)
    """
    if not np.ma.is_masked(gaia_dr3) and str(gaia_dr3).strip() not in ('', '--', '0'):
        return 2016.0
    if not np.ma.is_masked(gaia_dr2) and str(gaia_dr2).strip() not in ('', '--', '0'):
        return 2016.0
    if np.ma.is_masked(reflink) or not reflink:
        return None
    reflink_upper = reflink.upper()
    if any(k in reflink_upper for k in ('STASSUN', 'TICV', 'TIC_V')):
        return 2000.0
    if any(k in reflink_upper for k in ('HIPPARCOS', '_HIP_', 'HIC_')):
        return 1991.25
    pub_year = extract_year_from_reflink(reflink)
    if pub_year is not None:
        return 2016.0 if pub_year >= 2018 else float(pub_year)
    return None



class Crossmatcher:
    """
    Combines ID-based and coordinate-based crossmatching.
    
    First matches objects by ID, then performs coordinate-based
    matching on remaining unmatched sources.
    """
    
    def __init__(self):
        """
        Initialize crossmatcher.
        """
        # TODO: caching currently very primitive, should be improved
        # TODO: proper customization of keys and URLS
        self.catalogue = Table()
        self.catalogue_cached = False
        self.alternate_ids = Table()
        self.alternate_ids_cached = False
        self.id_matched = Table()
        self.coords2d_matched = Table()
        self.matched = Table()
        self.catalogue_suffix = "cat"
        self.input_suffix = "input"
        self.planet_uuid = "pl_name"
        self.input_starname_key = "star_name"
        self.catalogue_starname_key = "hostname"
        self.search_radius_arcsec = 10*u.arcsec

    def load_catalog(self, from_file=None, format="ascii") -> Table:
        if from_file is not None:
            self.catalogue = Table.read(from_file, format=format)
            self.catalogue_cached = True
            return self.catalogue
        
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        self.catalogue = nasa.run_sync(
            "SELECT * FROM pscomppars"
        ).to_table()
        self.catalogue_cached = True
        return self.catalogue

    def _expand_id_with_variants(self, input_id, id_str) -> list[tuple[str, str]]:
        """Expand an ID with possessive, NAME, and SIMBAD prefix variants."""
        id_str = id_str.strip()
        variants = [id_str]

        if id_str.startswith('*'):
            # Strip SIMBAD object type prefix: * = Star, ** = Star in double system (binary)
            # Per SIMBAD nomenclature, these prefixes appear in ~4800 IDs but NEA catalog doesn't use them
            variants.append(id_str.lstrip('* '))
        if id_str.endswith("'s"):
            # SIMBAD sometimes includes possessive forms of star names (Barnard's), but not the non-possessive form
            # we include the non-possessive form, because NEA uses it (Barnard b) 
            variants.append(id_str.rstrip("'s "))
        if id_str.startswith("NAME "):
            # SIMBAD sometimes includes "NAME " in front of star names (e.g. NAME Proxima Cen)
            # we include both versions
            variants.append(id_str.lstrip("NAME "))

        return [(input_id, v) for v in variants]


    def load_alternate_ids(self, name_list, from_file=None, format="ascii") -> Table:
        if from_file is not None:
            alternate_ids_file = Table.read(from_file, format=format)
            filtered = alternate_ids_file[np.isin(alternate_ids_file["input_ids"], name_list)]

            input_ids = []
            all_ids = []
            for row in filtered:
                input_id = str(row["input_ids"])
                id_str = str(row["id"])
                for _, variant_id in self._expand_id_with_variants(input_id, id_str):
                    input_ids.append(input_id)
                    all_ids.append(variant_id)

            self.alternate_ids = Table([input_ids, all_ids], names=["input_ids", "id"], dtype=["str", "str"])
            self.alternate_ids_cached = True
            return self.alternate_ids

        simbad = pyvo.dal.TAPService("https://simbad.cds.unistra.fr/simbad/sim-tap")
        alternate_ids_aggr = simbad.run_sync(
            "SELECT input_ids, ids FROM input_ids LEFT JOIN ident ON input_ids.input_ids = ident.id LEFT JOIN ids USING(oidref)",
        uploads={"input_ids": Table({"input_ids": name_list})}
        ).to_table()
        # SIMBAD's `ids` column is a single pipe-delimited string of all alternate identifiers
        # for an object (e.g. "Ross 128|GJ 447|HIP 57548|..."). We explode it here so each
        # identifier gets its own row, which is what id_crossmatch needs for exact-string joining.
        input_ids = []
        all_ids = []
        for row in alternate_ids_aggr:
            input_id = str(row["input_ids"])
            for id_str in str(row["ids"]).split("|"):
                for _, variant_id in self._expand_id_with_variants(input_id, id_str):
                    input_ids.append(input_id)
                    all_ids.append(variant_id)

        self.alternate_ids = Table([input_ids, all_ids], names=["input_ids", "id"], dtype=["str", "str"])
        self.alternate_ids_cached = True
        return self.alternate_ids

    def id_crossmatch(self, input_table):
        name_list = input_table[self.input_starname_key].tolist()
        if not self.alternate_ids_cached:
            self.load_alternate_ids(name_list)
        if not self.catalogue_cached:
            self.load_catalog()

        # Collapse runs of whitespace before joining. Catalogues don't agree on internal spacing:
        # Example: SIMBAD stores "Ross  128" (two spaces) while NEA stores "Ross 128" (one space), so an
        # exact-string join silently drops the match without this normalization.
        cat_df = self.catalogue.to_pandas()
        cat_df["hostname"] = cat_df["hostname"].str.split().str.join(" ")
        alt_df = self.alternate_ids.to_pandas()
        alt_df["id"] = alt_df["id"].str.split().str.join(" ")

        catalogue_projected_onto_ids = cat_df.merge(
            alt_df,
            left_on="hostname",
            right_on="id",
            how="inner"
        )

        self.id_matched = input_table.to_pandas().merge(
            catalogue_projected_onto_ids,
            left_on=self.input_starname_key,
            right_on="input_ids",
            how="right",
            suffixes=(f"_{self.input_suffix}", f"_{self.catalogue_suffix}")
        )

        self.id_matched["match_type"] = "id"
        self.id_matched = astropy.table.Table.from_pandas(self.id_matched)

        # When the merge result is empty, Astropy can infer string columns as float64.
        # Cast only actual string columns back to strings so vstack can merge correctly.
        string_columns = {
            name
            for name in self.catalogue.colnames
            if self.catalogue[name].dtype.kind in {"U", "S", "O"}
        } | {
            name
            for name in self.alternate_ids.colnames
            if self.alternate_ids[name].dtype.kind in {"U", "S", "O"}
        } | {self.input_starname_key, "match_type"}
        for colname in self.id_matched.colnames:
            if colname in string_columns and self.id_matched[colname].dtype.kind in {"f", "i"}:
                self.id_matched[colname] = self.id_matched[colname].astype("U")
                self.id_matched[colname] = np.char.strip(self.id_matched[colname]) 
                # we also strip whitespace in the process from any and all string columns
                # TODO: think about how sensible this is


        self.id_matched[self.input_starname_key] = self.id_matched[self.input_starname_key].astype(str)
        return self.id_matched

    def find_duplicates(self, input_table, full=False) -> Table:
        if not self.alternate_ids_cached:
            name_list = input_table[self.input_starname_key].tolist()
            self.load_alternate_ids(name_list)

        id_col = self.alternate_ids["id"]
        valid = (id_col != "") & (id_col != "--") # '--' is astropy masked representation
        alt_ids_without_nulls = self.alternate_ids[valid]
            
        self_joined = alt_ids_without_nulls.to_pandas().merge(
            alt_ids_without_nulls.to_pandas(),
            left_on="id",
            right_on="id",
            how="inner",
            suffixes=("", "_linked")
        )

        grouped: pd.DataFrame = self_joined.groupby('input_ids')['input_ids_linked'].unique().reset_index()
        grouped.rename(columns={"input_ids_linked": "duplicate_names"}, inplace=True)
        grouped['duplicate_names'] = grouped['duplicate_names'].apply(lambda x: list(sorted(x))) # cast to python list
        grouped['appearances'] = grouped['duplicate_names'].apply(len)
        grouped = grouped[grouped['appearances'] > 1]
        grouped = grouped.sort_values(by='duplicate_names').reset_index(drop=True)
        if full: return Table.from_pandas(grouped)
        
        grouped = grouped.drop_duplicates(subset='duplicate_names')
        return Table.from_pandas(grouped)
    
    
    def remove_duplicates(self, input_table) -> Table:
        """
        We keep the row with the highest number of non-null values
        """
        duplicates = self.find_duplicates(input_table)
        drop_indices = []
        for dupe in duplicates:
            dupe_names = dupe["duplicate_names"]
            # this will throw errors if duplicates is not found or two columns the same names are found
            dupes_index_comparison_array = np.array([input_table[self.input_starname_key].data == name for name in dupe_names])

            for arr, name in zip(dupes_index_comparison_array, dupe_names):
                if np.sum(arr) != 1:
                    raise ValueError(f"Expected exactly one match for each duplicate name, but found {np.sum(arr)} matches for {name}")
            
            # extract the indices of the duplicate rows
            dupe_indeces = np.where(dupes_index_comparison_array)[1]
            null_counts = [
                np.isin(list(input_table[idx]), ["", 'null', '0', 0, None]).sum() 
                for idx in dupe_indeces
            ]
    
            # we delete all but one of the duplicates, keeping the one with the least null values (but only one occurence)
            first_minimum_of_null = np.argmax(null_counts == min(null_counts))

            #TODO: decide what to do if there are multiple rows with the same number of null values

            drop_indices.extend(idx for i, idx in enumerate(dupe_indeces) if i != first_minimum_of_null)

        print(f"Removed Rows with indecies and names: {', '.join(str(i) for i in drop_indices)}")

        copy = input_table.copy()
        copy.remove_rows(drop_indices)
        return copy



    def coordinate_crossmatch(self, input_table, ra_key="ra", dec_key="dec") -> Table:
        if not self.catalogue_cached:
            self.load_catalog()

        epochs = [coord_epoch(rl, dr3, dr2) for rl, dr3, dr2 in
                  zip(self.catalogue['ra_reflink'], self.catalogue['gaia_dr3_id'], self.catalogue['gaia_dr2_id'])]
        self.catalogue['coord_epoch'] = MaskedColumn(
            np.ma.MaskedArray([e if e is not None else 0.0 for e in epochs],
                              mask=[e is None for e in epochs]),
            name='coord_epoch',
            description='Estimated epoch of sky coordinates (Julian year)')

        per_row_radius_2d = allowed_angular_seperation(
            self.catalogue["sy_pm"] / 1000,
            self.catalogue["sy_pmerr1"] / 1000,
            self.catalogue["coord_epoch"],
            minimum=self.search_radius_arcsec
        )

        coords_input = SkyCoord(ra=input_table[ra_key]*u.deg, dec=input_table[dec_key]*u.deg)
        coords_catalogue = SkyCoord(ra=self.catalogue["ra"]*u.deg, dec=self.catalogue["dec"]*u.deg)
        idx2d, sep2d, _  = coords_catalogue.match_to_catalog_sky(coords_input)
        sep2d_mask = sep2d < per_row_radius_2d

        self.coords2d_matched = astropy.table.hstack(
            [input_table[idx2d[sep2d_mask]], self.catalogue[sep2d_mask]],
            table_names=[self.input_suffix, self.catalogue_suffix],
            join_type="exact"
        )
        self.coords2d_matched["match_type"] = "2d"
        self.coords2d_matched["2d_sep"] = sep2d[sep2d_mask]
        return self.coords2d_matched

    def combined_crossmatch(self, input_table):
        uuid = self.planet_uuid

        id_results = self.id_crossmatch(input_table)
        coord_results = self.coordinate_crossmatch(input_table)

        only_coords = coord_results[
            ~np.isin(coord_results[uuid].tolist(), id_results[uuid].tolist())
        ]

        self.matched = astropy.table.vstack([id_results, only_coords], join_type="outer")
        return self.matched



if __name__ == "__main__":
    cm = Crossmatcher()
    input = Table.read("./input/HPIC_LC4_combined_d50.txt", format="ascii")
    cm.load_catalog(from_file="pscomppars.txt")
    cm.load_alternate_ids(input[cm.input_starname_key].tolist(), from_file="alternate_ids.txt")
    final = cm.combined_crossmatch(input)
    print(final)
