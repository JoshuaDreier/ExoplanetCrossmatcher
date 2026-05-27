import pytest
from astropy.table import Table, MaskedColumn, Column
from crossmatching import Crossmatcher


def make_catalog(*rows: dict[str, str | float]) -> Table:
    """
    Each row dict must have: hostname, pl_name and optionally ra, dec, sy_dist (each for all rows or for none).

    All proper-motion, distance-error, radial-velocity, and Gaia columns are
    masked, so add_coord_epoch_column defaults to epoch 2000.0 and the dynamic
    radii collapse to their minimum values (10 arcsec / 0.001 pc).
    """
    n = len(rows)

    def masked_col_float(val):
        return MaskedColumn([val] * n, mask=[True] * n, dtype=float)

    def masked_col_str(val, width=64):
        return MaskedColumn([val] * n, mask=[True] * n, dtype=f"U{width}")

    return Table({
        "hostname":    Column([r["hostname"] for r in rows]),
        "pl_name":     Column([r["pl_name"]  for r in rows]),
        "ra":          Column([r["ra"]        for r in rows], dtype=float) if "ra" in rows[0] else masked_col_float(0.0),
        "dec":         Column([r["dec"]       for r in rows], dtype=float) if "dec" in rows[0] else masked_col_float(0.0),
        "sy_dist":     Column([r["sy_dist"]   for r in rows], dtype=float) if "sy_dist" in rows[0] else masked_col_float(0.0),
        "sy_pm":       masked_col_float(0.0),
        "sy_pmerr1":   masked_col_float(0.0),
        "sy_disterr1": masked_col_float(0.0),
        "sy_disterr2": masked_col_float(0.0),
        "st_radv":     masked_col_float(0.0),
        "sy_gaiamag":  masked_col_float(10.0),
        "gaia_dr3_id": masked_col_str(""),
        "gaia_dr2_id": masked_col_str(""),
        "ra_reflink":  masked_col_str("", width=256),
    })


@pytest.fixture
def fresh_cm():
    return Crossmatcher()
