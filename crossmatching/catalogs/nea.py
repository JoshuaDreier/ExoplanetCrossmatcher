import re
import numpy as np
import pyvo
import astropy.units as u
from astropy.table import Table, MaskedColumn

from crossmatching.catalogs.base import CatalogBase


def _extract_year_from_reflink(reflink):
    """Parse a publication year from an ADS-style HTML reflink string."""
    if not reflink or np.ma.is_masked(reflink):
        return None
    match = re.search(r'refstr=["\']?[A-Z_]+_(\d{4})', reflink)
    if match:
        return int(match.group(1))
    match = re.search(r'\b(19|20)\d{2}\b', reflink)
    if match:
        return int(match.group(0))
    return None


def _coord_epoch(reflink, gaia_dr3, gaia_dr2):
    """
    Estimate the coordinate epoch for a single NEA catalog row.
    Returns a float (Julian year) or None if unknown.

    Priority:
      1. Gaia DR3 id present  -> 2016.0
      2. Gaia DR2 id present  -> 2016.0
      3. ra_reflink mentions TICv8/Stassun -> 2000.0
      4. ra_reflink mentions Hipparcos     -> 1991.25
      5. ra_reflink publication year >= 2018 -> 2016.0
      6. ra_reflink publication year < 2018  -> that year
      7. Default -> None
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
    pub_year = _extract_year_from_reflink(reflink)
    if pub_year is not None:
        return 2016.0 if pub_year >= 2018 else float(pub_year)
    return None


class NEACatalog(CatalogBase):
    """Catalog adapter for the NASA Exoplanet Archive Planetary Systems table.

    Downloads the ``pscomppars`` composite-parameters table via TAP and
    adds a ``coord_epoch`` column estimated from each row's Gaia cross-ID
    and coordinate reference link.

    Attributes
    ----------
    ra_key : str
        ``'ra'`` right ascension column (mixed epochs)
    ra_unit: u.Unit
        right ascension unit, degrees, u.deg
    dec_key : str
        ``'dec'`` declination column (mixed epochs)
    dec_unit: u.Unit
        declination unit, degrees, u.deg
    host_key : str
        ``'hostname'`` — host-star name column (join key for ID matching).
    planet_uuid : str
        ``'pl_name'`` — unique planet-name column.
    pm_key : str
        ``'sy_pm'`` — total proper-motion column (mas/yr).
    pmerr_key : str
        ``'sy_pmerr1'`` — proper-motion uncertainty column (mas/yr).
    """

    ra_key = "ra"
    ra_unit = u.deg
    dec_key = "dec"
    dec_unit = u.deg
    host_key = "hostname"
    planet_uuid = "pl_name"
    pm_key = "sy_pm"
    pmerr_key = "sy_pmerr1"

    def download(self) -> Table:
        """Query the NASA Exoplanet Archive TAP and return the raw table.

        Fetches ``SELECT * FROM pscomppars`` from the TAP endpoint at
        ``exoplanetarchive.ipac.caltech.edu``.  Requires a network
        connection; use :meth:`~CatalogBase.load` with ``from_file=``
        for offline use.

        Returns
        -------
        table : `~astropy.table.Table`
            Raw ``pscomppars`` table with no preprocessing applied.
        """
        nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        return nasa.run_sync("SELECT * FROM pscomppars").to_table()

    def preprocess(self, table: Table) -> Table:
        """Add ``coord_epoch`` to the NASA Exoplanet Archive table.

        Estimates the Julian-year epoch of each row's sky coordinates
        using the following priority order:

        1. Gaia DR3 or DR2 cross-ID present → 2016.0
        2. ``ra_reflink`` mentions TICv8 / Stassun → 2000.0
        3. ``ra_reflink`` mentions Hipparcos → 1991.25
        4. ``ra_reflink`` publication year ≥ 2018 → 2016.0
        5. ``ra_reflink`` publication year < 2018 → that year as float
        6. None of the above → masked (coordinate matching falls back to
           ``unknown_default`` = 50 arcsec [default, but configurable] for these rows)

        Parameters
        ----------
        table : `~astropy.table.Table`
            Raw table from :meth:`download` or :meth:`~CatalogBase.load_raw`.

        Returns
        -------
        table : `~astropy.table.Table`
            Input table with one additional column:

            - ``coord_epoch`` : `~astropy.table.MaskedColumn` of float,
              estimated coordinate epoch (Julian year).  Masked where
              the epoch cannot be determined.
        """
        epochs = [
            _coord_epoch(rl, dr3, dr2)
            for rl, dr3, dr2 in zip(
                table['ra_reflink'],
                table['gaia_dr3_id'],
                table['gaia_dr2_id'],
            )
        ]
        table['coord_epoch'] = MaskedColumn(
            np.ma.MaskedArray(
                [e if e is not None else 0.0 for e in epochs],
                mask=[e is None for e in epochs],
            ),
            name='coord_epoch',
            description='Estimated epoch of sky coordinates (Julian year)',
        )
        return table
