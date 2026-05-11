from typing import Tuple
from unicodedata import name
import astropy
import numpy as np
import pyvo
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


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
        self.search_radius_arcsec = 15*u.arcsec
        self.search_radius_pc = 0.005*u.pc # corresponds to around 82 arcsec at 10 pc 
    
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
            self.alternate_ids = Table.read(from_file, format=format)
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
        for input_id, alt_ids in alternate_ids_aggr.iterrows():
            for id in str(alt_ids).split("|"):
                input_ids.append(input_id)
                all_ids.append(id)
                # all_ids.append(id.replace("'", "''")) # in ADQL, escape single quote ' with ''

        self.alternate_ids = Table([input_ids, all_ids], names=["input_ids", "id"], dtype=["str", "str"])
        self.alternate_ids_cached = True
        return self.alternate_ids

    def id_crossmatch(self, input_table):
        name_list = input_table[self.input_starname_key].tolist()
        if not self.alternate_ids_cached:
            self.load_alternate_ids(name_list)
        if not self.catalogue_cached:
            self.load_catalog()

        catalogue_projected_onto_ids = \
            self.catalogue.to_pandas().merge(
                self.alternate_ids.to_pandas(),
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
                np.isin(list(input_table[idx]), ["", 'null', '0', 0]).sum() 
                for idx in dupe_indeces
            ]
    
            # we delete all but one of the duplicates, keeping the one with the least null values (but only one occurence)
            first_minimum_of_null = np.argmin(null_counts == min(null_counts))

            #TODO: decide what to do if there are multiple rows with the same number of null values

            drop_indices.extend(idx for i, idx in enumerate(dupe_indeces) if i != first_minimum_of_null)

        print(f"Removed Rows with indecies and names: {', '.join(str(i) for i in drop_indices)}")

        copy = input_table.copy()
        copy.remove_rows(drop_indices)
        return copy



    def coordinate_crossmatch(self, input_table, ra_key="ra", dec_key="dec", distance_key="sy_dist") -> Tuple[Table]:
        if not self.catalogue_cached:
            self.load_catalog()

        coords_input = SkyCoord(
            ra=input_table[ra_key]*u.deg,
            dec=input_table[dec_key]*u.deg,
            distance=input_table[distance_key]*u.pc
        )
        coords_catalogue = SkyCoord(
            ra=self.catalogue["ra"]*u.deg,
            dec=self.catalogue["dec"]*u.deg, 
            distance=(self.catalogue["sy_dist"]*u.pc).to(u.pc)
        )

        idx2d, sep2d, _ = coords_catalogue.match_to_catalog_sky(coords_input)
        idx3d, _, sep3d = coords_catalogue.match_to_catalog_3d(coords_input)
        sep2d_mask = sep2d < self.search_radius_arcsec
        sep3d_mask = sep3d < self.search_radius_pc
        self.coords3d_matched = astropy.table.hstack(
            [input_table[idx3d[sep3d_mask]], self.catalogue[sep3d_mask]],
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
