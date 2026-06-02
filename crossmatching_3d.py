# Legacy 3D coordinate crossmatching — kept for notebook use.
# Notebooks that need 3D matching should import Crossmatcher3D from here.
import numpy as np
import astropy
import astropy.units as u
from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord

from crossmatching import Crossmatcher, allowed_angular_separation


def allowed_3d_separation(angular_radius_2d, radial_velocity, dist_pc, mean_dist_err,
                           gaia_mag, epoch, hpic_epoch=2000, minimum=0.001*u.pc):
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
    gaia_bias = 0
    total = np.sqrt(transverse**2 + radial**2 + dist_unc**2) + gaia_bias
    return total + minimum


class Crossmatcher3D(Crossmatcher):
    """
    Crossmatcher with legacy 3D coordinate matching.
    coordinate_crossmatch() returns (3d_result, 2d_result) as before.
    """

    def __init__(self, input_starname_key: str):
        from crossmatching import NEACatalog, SimbadIdSupplier
        super().__init__(NEACatalog(), SimbadIdSupplier(), input_starname_key=input_starname_key)
        self.coords3d_matched = Table()
        self.search_radius_pc = 0.001 * u.pc

    def coordinate_crossmatch(self, input_table, ra_key="ra", dec_key="dec",
                               distance_key="sy_dist"):
        if self.catalog_table is None:
            self.load_catalog()

        if 'coord_epoch' in self.catalog_table.colnames:
            epoch = self.catalog_table['coord_epoch']
        else:
            n = len(self.catalog_table)
            epoch = MaskedColumn(
                np.ma.MaskedArray(np.zeros(n), mask=np.ones(n, dtype=bool)),
                name='coord_epoch',
            )

        per_row_radius_2d = allowed_angular_separation(
            self.catalog_table["sy_pm"] / 1000,
            self.catalog_table["sy_pmerr1"] / 1000,
            epoch,
            minimum=self.coordinate_search_radius
        )

        # 2D matching (via parent's coordinate_crossmatch)
        super().coordinate_crossmatch(input_table, ra_key, dec_key)

        # 3D matching
        mean_dist_err = (self.catalog_table["sy_disterr1"] - self.catalog_table["sy_disterr2"]).filled(0) / 2

        has_distance_input     = input_table[distance_key] > 0
        has_distance_catalog = self.catalog_table["sy_dist"] > 0

        input_3d = input_table[has_distance_input]
        cat_3d   = self.catalog_table[has_distance_catalog]

        coords_input_3d = SkyCoord(
            ra=input_3d[ra_key]*u.deg, dec=input_3d[dec_key]*u.deg,
            distance=input_3d[distance_key]*u.pc,
        )
        coords_cat_3d = SkyCoord(
            ra=cat_3d["ra"]*u.deg, dec=cat_3d["dec"]*u.deg,
            distance=cat_3d["sy_dist"]*u.pc,
        )
        per_row_radius_3d = allowed_3d_separation(
            per_row_radius_2d[has_distance_catalog],
            cat_3d["st_radv"].filled(0),
            cat_3d["sy_dist"],
            mean_dist_err[has_distance_catalog],
            cat_3d["sy_gaiamag"].filled(np.inf),
            epoch[has_distance_catalog],
            minimum=self.search_radius_pc
        )
        idx3d, _, sep3d = coords_cat_3d.match_to_catalog_3d(coords_input_3d)
        sep3d_mask = sep3d < per_row_radius_3d

        # idx3d indexes into the filtered input_3d; remap to original input_table positions
        input_valid_idx = np.where(has_distance_input)[0]

        self.coords3d_matched = astropy.table.hstack(
            [input_table[input_valid_idx[idx3d[sep3d_mask]]], cat_3d[sep3d_mask]],
            table_names=[self.input_suffix, ""],
            join_type="exact"
        )
        self.coords3d_matched["match_type"] = "3d"
        self.coords3d_matched["3d_sep"] = sep3d[sep3d_mask]

        return (self.coords3d_matched, self.coords2d_matched)

    def combined_crossmatch(self, input_table):
        uuid = self.planet_uuid

        id_results = self.id_crossmatch(input_table)
        matches_3d, matches_2d = self.coordinate_crossmatch(input_table)

        id_planets = set(id_results[uuid].tolist())

        only_3d = matches_3d[~np.isin(matches_3d[uuid].tolist(), list(id_planets))]

        already_found = id_planets | set(only_3d[uuid].tolist())
        only_2d = matches_2d[~np.isin(matches_2d[uuid].tolist(), list(already_found))]

        self.matched = astropy.table.vstack([id_results, only_3d, only_2d], join_type="outer")
        return self.matched
