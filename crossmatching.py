from typing import Tuple
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
    
    def load_catalog(self, from_file=None, format="ascii") -> Table:
        if from_file is not None:
            self.catalogue = Table.read(from_file, format=format)
            self.catalogue_cached = True
            return self.catalogue
        
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        self.catalogue = nasa.run_sync(
            "SELECT * FROM pscomppars"
        )
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
            self.get_alternate_id_list(name_list)
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
                

        return self.id_matched

    def coordinate_crossmatch(self, input_table, ra_key="ra", dec_key="dec", distance_key="sy_dist", radius_arcsec=15, radius_pc=0.02) -> Tuple[Table]:
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

        idx, sep2d, sep3d = coords_input.match_to_catalog_sky(coords_catalogue)
        sep2d_mask = sep2d < radius_arcsec*u.arcsec
        sep3d_mask = sep3d < radius_pc*u.pc
        self.coords3d_matched = astropy.table.hstack(
            [input_table[sep3d_mask], self.catalogue[idx[sep3d_mask]], Table([["3d"]*sum(sep3d_mask)], names=["match_type"])],
            table_names=[self.input_suffix, self.catalogue_suffix, "match_type"],
            join_type="exact"
        )
        self.coords3d_matched["3d_sep"] = sep3d[sep3d_mask]
        self.coords2d_matched = astropy.table.hstack(
            [input_table[sep2d_mask], self.catalogue[idx[sep2d_mask]], Table([["2d"]*sum(sep2d_mask)], names=["match_type"])],
            table_names=[self.input_suffix, self.catalogue_suffix, "match_type"],
            join_type="exact"

        )    
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
