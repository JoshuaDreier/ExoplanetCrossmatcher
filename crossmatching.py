from typing import Tuple
from unicodedata import name
import astropy
import numpy as np
import pyvo
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Column
import re


def allowed_3d_seperation(angular_radius_2d, radial_velocity, dist_pc, mean_dist_err, gaia_mag, epoch, hpic_epoch=2000, minimum=0.001*u.pc):
    """
    Estimate the maximum 3D spatial displacement (pc) of a source since hpic_epoch.

    angular_radius_2d: already-floored 2D threshold (astropy Quantity, arcsec)
    radial_velocity:   radial velocity in km/s (array OK; use 0 for unknown)
    dist_pc:           distance in parsecs (array OK)
    mean_dist_err:     mean distance uncertainty in parsecs (array OK; use 0 for unknown)
    gaia_mag:          Gaia G-band magnitude (array OK; use np.inf for unknown/dim stars)
    epoch:             estimated coordinate epoch (array OK)
    minimum:           floor (astropy Quantity with pc unit)
    """
    dt = np.abs(epoch - hpic_epoch) * u.yr
    transverse = angular_radius_2d.to(u.rad).value * dist_pc * u.pc
    radial = (np.abs(radial_velocity) * u.km / u.s * dt).to(u.pc)
    dist_unc = mean_dist_err * u.pc
    # Gaia DR2 parallax systematic for stars with G < 5 (saturation/calibration bias).
    # δd = δπ × d²/1000 with δπ ≈ 2 mas 
    # gaia_bias = np.where(
    #     gaia_mag < 5,
    #     2*1e-3*dist_pc**2,
    #     0.0
    # ) * u.pc
    gaia_bias = 0
    total = np.sqrt(transverse**2 + radial**2 + dist_unc**2) + gaia_bias
    return total + minimum


def allowed_angular_seperation(proper_motion, pm_err, epoch, hpic_epoch=2000, minimum=10*u.arcsec):
    """inputs must be in arcsec/yr"""
    dt = np.abs(epoch - hpic_epoch)
    return ((proper_motion + pm_err) * dt)*u.arcsec + minimum



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


def add_coord_epoch_column(table, col_name='coord_epoch'):
    """
    Estimate the epoch of the sky coordinates for each row and add it
    as a new column to the table.

    Priority:
      1. Gaia DR3 id present  -> 2016.0
      2. Gaia DR2 id present  -> 2016.0
      3. ra_reflink mentions TICv8/Stassun -> 2000.0
      4. ra_reflink mentions Hipparcos     -> 1991.25
      5. ra_reflink publication year >= 2018 (post-Gaia DR2) -> 2016.0
      6. ra_reflink publication year < 2018  -> that year (rough proxy)
      7. Default -> 2000.0
    """
    n = len(table)
    epochs = np.full(n, 2000.0)  # default

    # Work column-by-column — faster than row-by-row per astropy docs
    gaia_dr3 = table['gaia_dr3_id']
    gaia_dr2 = table['gaia_dr2_id']
    reflinks = table['ra_reflink']

    for i in range(n):
        dr3 = gaia_dr3[i]
        dr2 = gaia_dr2[i]
        reflink = reflinks[i]

        # 1 & 2: Gaia ID is the most reliable epoch indicator
        if not np.ma.is_masked(dr3) and str(dr3).strip() not in ('', '--', '0'):
            epochs[i] = 2016.0
            continue
        if not np.ma.is_masked(dr2) and str(dr2).strip() not in ('', '--', '0'):
            epochs[i] = 2016.0
            continue

        # 3–6: parse the reflink
        if np.ma.is_masked(reflink) or not reflink:
            continue  # keep default 2000.0

        reflink_upper = reflink.upper()

        if any(k in reflink_upper for k in ('STASSUN', 'TICV', 'TIC_V')):
            epochs[i] = 2000.0
            continue

        if any(k in reflink_upper for k in ('HIPPARCOS', '_HIP_', 'HIC_')):
            epochs[i] = 1991.25
            continue

        pub_year = extract_year_from_reflink(reflink)
        if pub_year is not None:
            epochs[i] = 2016.0 if pub_year >= 2018 else float(pub_year)

    table[col_name] = Column(epochs, name=col_name,
                             description='Estimated epoch of sky coordinates (Julian year)')
    return table


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
        self.coords3d_matched = Table()
        self.coords2d_matched = Table()
        self.matched = Table()  
        self.catalogue_suffix = "cat"
        self.input_suffix = "input"
        self.planet_uuid = "pl_name"
        self.input_starname_key = "star_name"
        self.catalogue_starname_key = "hostname"
        self.search_radius_arcsec = 10*u.arcsec
        # self.search_radius_pc = 0.05 * u.pc # corresponds to around 160 arcsec at 10 pc 
        self.search_radius_pc = 0.001*u.pc
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

    def load_alternate_ids(self, name_list, from_file=None, format="ascii") -> Table:
        if from_file is not None:
            alternate_ids_file = Table.read(from_file, format=format)
            self.alternate_ids = alternate_ids_file[np.isin(alternate_ids_file["input_ids"], name_list)]
            self.alternate_ids_cached = True
            return self.alternate_ids
        simbad = pyvo.dal.TAPService("https://simbad.cds.unistra.fr/simbad/sim-tap")
        alternate_ids_aggr = simbad.run_sync(
            "SELECT input_ids, ids FROM input_ids LEFT JOIN ident ON input_ids.input_ids = ident.id LEFT JOIN ids USING(oidref)",
        uploads={"input_ids": Table({"input_ids": name_list})}
        ).to_table()
        # takes around 1m30s
        input_ids = []
        all_ids = []
        # SIMBAD's `ids` column is a single pipe-delimited string of all alternate identifiers
        # for an object (e.g. "Ross 128|GJ 447|HIP 57548|..."). We explode it here so each
        # identifier gets its own row, which is what id_crossmatch needs for exact-string joining.
        for row in alternate_ids_aggr:
            for id in str(row["ids"]).split("|"):
                input_ids.append(str(row["input_ids"]))
                all_ids.append(id)

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

        alt_ids_without_nulls = self.alternate_ids[~self.alternate_ids["id"].mask]
            
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



    def coordinate_crossmatch(self, input_table, ra_key="ra", dec_key="dec", distance_key="sy_dist") -> Tuple[Table]:
        if not self.catalogue_cached:
            self.load_catalog()

        add_coord_epoch_column(self.catalogue)

        per_row_radius_2d = allowed_angular_seperation(
            self.catalogue["sy_pm"].filled(0) / 1000,
            self.catalogue["sy_pmerr1"].filled(0) / 1000,
            self.catalogue["coord_epoch"],
            minimum=self.search_radius_arcsec
        )
        mean_dist_err = (self.catalogue["sy_disterr1"] - self.catalogue["sy_disterr2"]).filled(0) / 2

        # 2D sky matching — distance irrelevant for angular separation
        coords_input     = SkyCoord(ra=input_table[ra_key]*u.deg, dec=input_table[dec_key]*u.deg)
        coords_catalogue = SkyCoord(ra=self.catalogue["ra"]*u.deg, dec=self.catalogue["dec"]*u.deg)
        idx2d, sep2d, _  = coords_catalogue.match_to_catalog_sky(coords_input)
        sep2d_mask = sep2d < per_row_radius_2d

        # 3D spatial matching — skip rows with zero or missing distance to avoid
        # degenerate matches where both stars collapse to the 3D origin
        has_distance_input = input_table[distance_key] > 0
        has_distance_catalogue = self.catalogue["sy_dist"] > 0

        input_3d = input_table[has_distance_input]
        cat_3d = self.catalogue[has_distance_catalogue]
        coords_input_3d = SkyCoord(
            ra=input_3d[ra_key]*u.deg, dec=input_3d[dec_key]*u.deg,
            distance=input_3d[distance_key]*u.pc,
        )
        coords_cat_3d = SkyCoord(
            ra=cat_3d["ra"]*u.deg, dec=cat_3d["dec"]*u.deg,
            distance=cat_3d["sy_dist"]*u.pc,
        )
        per_row_radius_3d = allowed_3d_seperation(
            per_row_radius_2d[has_distance_catalogue],
            cat_3d["st_radv"].filled(0),
            cat_3d["sy_dist"],
            mean_dist_err[has_distance_catalogue],
            cat_3d["sy_gaiamag"].filled(np.inf),
            cat_3d["coord_epoch"],
            minimum=self.search_radius_pc
        )
        idx3d, _, sep3d = coords_cat_3d.match_to_catalog_3d(coords_input_3d)
        sep3d_mask = sep3d < per_row_radius_3d

        # idx3d indexes into input_3d; map back to original input_table row positions
        input_valid_idx = np.where(has_distance_input)[0]

        self.coords3d_matched = astropy.table.hstack(
            [input_table[input_valid_idx[idx3d[sep3d_mask]]], cat_3d[sep3d_mask]],
            table_names=[self.input_suffix, self.catalogue_suffix],
            join_type="exact"
        )
        self.coords3d_matched["match_type"] = "3d"
        self.coords3d_matched["3d_sep"] = sep3d[sep3d_mask]

        self.coords2d_matched = astropy.table.hstack(
            [input_table[idx2d[sep2d_mask]], self.catalogue[sep2d_mask]],
            table_names=[self.input_suffix, self.catalogue_suffix],
            join_type="exact"
        )
        self.coords2d_matched["match_type"] = "2d"
        self.coords2d_matched["2d_sep"] = sep2d[sep2d_mask]
        return (self.coords3d_matched, self.coords2d_matched)

    def combined_crossmatch(self, input_table):
        # TODO: key debugging so output format matches
        # TODO: proper keys passing (in both senses of the word)
        # self.id_crossmatch(input_table)
        # self.coordinate_crossmatch(input_table)
        
        if len(self.id_matched) == 0:
            self.id_crossmatch(input_table)
        if len(self.coords3d_matched) == 0 or len(self.coords2d_matched) == 0:
            self.coordinate_crossmatch(input_table)        
        
        uuid = self.planet_uuid

        matched_listid = self.id_matched[uuid].tolist()
        matched_list3d = self.coords3d_matched[uuid].tolist()
        matched_list2d = self.coords2d_matched[uuid].tolist()
    
        only_coords3d = self.coords3d_matched[
            ~np.isin(matched_list3d, matched_listid)
        ]
        only_coords2d = self.coords2d_matched[
            ~(np.isin(matched_list2d, matched_listid)
            | np.isin(matched_list2d, only_coords3d[uuid].tolist()))
        ]

        self.matched = astropy.table.vstack([self.id_matched, only_coords3d, only_coords2d], join_type="outer")
        return self.matched



if __name__ == "__main__":
    cm = Crossmatcher()
    input = Table.read("./input/HPIC_LC4_combined_d50.txt", format="ascii")
    cm.load_catalog(from_file="pscomppars.txt")
    cm.load_alternate_ids(input[cm.input_starname_key].tolist(), from_file="alternate_ids.txt")
    final = cm.combined_crossmatch(input)
    print(final)
