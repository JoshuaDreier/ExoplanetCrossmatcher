import numpy as np
import pyvo
from astropy.table import Table

from crossmatching.id_suppliers.base import IdSupplierBase


class SimbadIdSupplier(IdSupplierBase):
    """Alternate-ID supplier backed by the SIMBAD TAP service.

    Queries SIMBAD for all known identifiers of each input star via an
    uploaded-table TAP query, then expands the pipe-delimited ``ids``
    column into individual (star, identifier) pairs.  Four variant forms
    are generated per raw ID to bridge naming discrepancies between
    SIMBAD and planet catalogs:

    1. Strip leading ``*`` / ``**`` object-type prefixes
       (e.g. ``'* alf Cen'`` → ``'alf Cen'``)
    2. Strip trailing ``'s`` possessive suffix
       (e.g. ``"Barnard's"`` → ``'Barnard'``)
    3. Strip leading ``'NAME '`` prefix
       (e.g. ``'NAME Proxima Cen'`` → ``'Proxima Cen'``)
    4. Drop trailing stellar component letter (space + A/B/C/S/N)
       (e.g. ``'Kepler-1229 A'`` → ``'Kepler-1229'``)

    The ``input_col``, ``id_col``, and ``null_sentinel`` values are
    inherited from :class:`IdSupplierBase` (``'input_ids'``, ``'id'``,
    ``'--'``).
    """

    def _expand_id_with_variants(self, input_id, id_str):
        """Return (input_id, variant) pairs for a single raw SIMBAD identifier.

        Generates the original ID plus up to four additional variants to
        bridge naming differences between SIMBAD and planet catalogs.
        See class docstring for the full list of rules.

        Parameters
        ----------
        input_id : str
            The input star name this identifier belongs to.
        id_str : str
            A single identifier string from SIMBAD (may have leading/
            trailing whitespace).

        Returns
        -------
        pairs : list of (str, str)
            List of ``(input_id, variant)`` tuples.  The first element
            is always ``(input_id, id_str.strip())``.
        """
        id_str = id_str.strip()
        variants = [id_str]

        if id_str.startswith('*'):
            # Strip SIMBAD object type prefix: * = Star, ** = Star in double system (binary)
            # Per SIMBAD nomenclature, these prefixes appear in ~4800 IDs but NEA catalog doesn't use them
            variants.append(id_str.lstrip('* '))
        if id_str.endswith("'s"):
            # SIMBAD sometimes includes possessive forms of star names (Barnard's), but not the non-possessive form
            # we include the non-possessive form, because NEA uses it (Barnard b)
            variants.append(id_str[:-2])
        if id_str.startswith("NAME "):
            # SIMBAD sometimes includes "NAME " in front of star names (e.g. NAME Proxima Cen)
            # we include both versions
            variants.append(id_str[5:])
        if len(id_str) > 2 and id_str[-2] == ' ' and id_str[-1] in 'ABCSN':
            # Some catalogs append a stellar component letter (e.g. 'Kepler-1229 A') while
            # others use the bare host name ('Kepler-1229'); include both forms
            variants.append(id_str[:-2])

        return [(input_id, v) for v in variants]

    def download(self, name_list: list[str]) -> Table:
        """Query SIMBAD TAP for all alternate identifiers of the input stars.

        Uploads the name list as a table and runs a LEFT JOIN across
        SIMBAD's ``input_ids``, ``ident``, and ``ids`` tables.  Stars
        not found in SIMBAD appear in the result with a null ``ids``
        value.

        Parameters
        ----------
        name_list : list of str
            Input star names to look up.

        Returns
        -------
        raw : `~astropy.table.Table`
            Two-column table: ``input_col`` (input name) and ``'ids'``
            (pipe-delimited SIMBAD identifiers, or null senti        if len(id_str) > 2 and id_str[-2] == ' ' and id_str[-1] in 'ABCSN':
            # Some catalogs append a stellar component letter (e.g. 'Kepler-1229 A') while
            # others use the bare host name ('Kepler-1229'); include both forms
            variants.append(id_str[:-2])nel when
            no match is found).
        """
        simbad = pyvo.dal.TAPService("https://simbad.cds.unistra.fr/simbad/sim-tap")
        return simbad.run_sync(
            "SELECT input_ids, ids FROM input_ids LEFT JOIN ident ON input_ids.input_ids = ident.id LEFT JOIN ids USING(oidref)",
            uploads={"input_ids": Table({self.input_col: name_list})}
        ).to_table()

    def load_alternate_ids(self, name_list: list[str], from_file: str = None, format: str = "ascii") -> Table:
        """Load alternate identifiers, filtered to the requested names.

        Overrides the base-class method to add a subsetting step when
        reading from a cached file: the file may contain IDs for a
        superset of ``name_list``, so rows for names not in
        ``name_list`` are dropped before preprocessing.

        Parameters
        ----------
        name_list : list of str
            Input star names to load IDs for.
        from_file : str, optional
            Path to a previously saved raw alternate-ID file.  If given,
            reads from disk (no network call); otherwise calls
            :meth:`download`.
        format : str, optional
            Astropy table format string (default ``'ascii'``).

        Returns
        -------
        ids : `~astropy.table.Table`
            Two-column string table: ``input_col`` and ``id_col``.
            Contains only rows whose ``input_col`` value appears in
            ``name_list``.
        """
        raw = self.load_raw(from_file, format=format) if from_file else self.download(name_list)
        if from_file:
            raw = raw[np.isin(raw[self.input_col], name_list)]
        return self.preprocess(raw)

    def preprocess(self, raw: Table) -> Table:
        """Expand pipe-delimited SIMBAD IDs into one-row-per-identifier form.

        Iterates over the raw SIMBAD query result, splitting each
        ``ids`` field on ``'|'`` and calling
        :meth:`_expand_id_with_variants` on each token.  Empty strings
        and null-sentinel entries (``'--'``) are discarded.

        Parameters
        ----------
        raw : `~astropy.table.Table`
            Raw table from :meth:`download`.  Expected columns:
            ``input_col`` (input star name) and ``'ids'`` (pipe-delimited
            SIMBAD identifiers).

        Returns
        -------
        ids : `~astropy.table.Table`
            Two-column string table: ``input_col`` and ``id_col``.
            One row per (star, identifier variant) pair; a single input
            star may contribute many rows.
        """
        input_ids = []
        all_ids = []
        for row in raw:
            input_id = str(row[self.input_col])
            for id_str in str(row["ids"]).split("|"):
                stripped = id_str.strip()
                if not stripped or stripped == self.null_sentinel:
                    continue
                for _, variant_id in self._expand_id_with_variants(input_id, id_str):
                    input_ids.append(input_id)
                    all_ids.append(variant_id)
        return Table([input_ids, all_ids], names=[self.input_col, self.id_col], dtype=["str", "str"])
