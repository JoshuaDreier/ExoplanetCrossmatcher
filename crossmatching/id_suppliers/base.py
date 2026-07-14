from astropy.table import Table


class IdSupplierBase:
    """Abstract base class for alternate-ID suppliers.

    An IdSupplier maps each input star name to zero or more alternate
    identifiers drawn from an external catalog (e.g. SIMBAD).  These
    alternate IDs are then joined against the planet catalog's host-name
    column to perform ID-based crossmatching.

    Subclasses must implement :meth:`download` and, optionally,
    :meth:`preprocess` to expand or normalise raw ID strings.

    Attributes
    ----------
    input_col : str
        Column name for input star names in the alternate-ID table
        (default ``'input_ids'``; override by setting the attribute on a
        subclass or instance).
    id_col : str
        Column name for alternate identifiers in the alternate-ID table
        (default ``'id'``).
    null_sentinel : str
        Placeholder string meaning "no ID available" (default ``'--'``).
        Rows with this value in ``id_col`` are skipped during matching.
    """

    input_col:     str = "input_ids"
    id_col:        str = "id"
    null_sentinel: str = "--"

    def id_variants(self, id_str: str) -> list[str]:
        """Return the match forms of one identifier: itself plus variants.

        Each rule bridges a known naming discrepancy between SIMBAD-style
        identifiers and planet-catalog host names.  All rules are applied
        to the original ``id_str`` only (single-step, not transitive).
        The one cross-rule composition that is useful —
        ``"NAME Barnard's"`` → ``"Barnard"`` — is generated explicitly
        inside the ``NAME`` rule.

        Identifiers differing by a trailing stellar component letter
        (``GJ 86 B`` vs ``GJ 86``) are deliberately *not* bridged: on
        current data this adds no matches for the nea-simbad and emc-emc
        configs while attaching planets to the wrong binary component in
        emc-simbad (see ``companion_mismatch.ipynb``).

        Subclasses extend the list by overriding this method and calling
        ``super().id_variants()``.

        Parameters
        ----------
        id_str : str
            A single identifier string (may have leading/trailing
            whitespace).

        Returns
        -------
        variants : list of str
            Stripped, deduplicated, non-empty forms; the original
            (stripped) ``id_str`` is always first.
        """
        id_str = id_str.strip()
        variants = [id_str]
        if id_str.startswith('*'):
            # Strip SIMBAD object type prefix: * = Star, ** = Star in double system (binary)
            # Per SIMBAD nomenclature, these prefixes appear in ~4800 IDs but planet catalogs don't use them
            variants.append(id_str.lstrip('* '))
        if id_str.startswith("NAME "):
            # SIMBAD sometimes includes "NAME " in front of star names (e.g. NAME Proxima Cen)
            variants.append(id_str[5:])
        if id_str.endswith("'s"):
            # SIMBAD sometimes includes possessive forms of star names (Barnard's), but not the
            # non-possessive form; planet catalogs use the non-possessive form (Barnard b)
            variants.append(id_str[:-2])
            if id_str.startswith("NAME "):
                # Compound: "NAME Barnard's" → "Barnard" (the one useful two-rule composition)
                variants.append(id_str[5:-2])
        return [v for v in dict.fromkeys(v.strip() for v in variants) if v]

    def download(self, name_list: list[str]) -> Table:
        """Query the remote source and return the raw Table."""
        raise NotImplementedError

    def save_raw(self, name_list: list[str], path: str, format: str = "ascii") -> None:
        """Download raw data and write it to a file."""
        self.download(name_list).write(path, format=format, overwrite=True)

    def load_raw(self, path: str, format: str = "ascii") -> Table:
        """Read a raw file without preprocessing."""
        return Table.read(path, format=format)

    def preprocess(self, raw: Table) -> Table:
        """Apply source-specific transformations to a raw ID table.

        Default implementation is a no-op.  Subclasses override to expand
        pipe-delimited ID strings into one-row-per-ID form and to generate
        name variants (e.g. stripping SIMBAD prefixes).

        Parameters
        ----------
        raw : `~astropy.table.Table`
            Raw table returned by :meth:`download` or :meth:`load_raw`.
            Expected columns: ``input_col``, plus a raw ID column.

        Returns
        -------
        ids : `~astropy.table.Table`
            Normalised table with exactly two string columns:
            ``input_col`` and ``id_col``.
        """
        return raw

    def load_alternate_ids(
        self,
        name_list: list[str],
        from_file: str | None = None,
        format: str = "ascii",
    ) -> Table:
        """Load alternate identifiers and apply preprocessing.

        Parameters
        ----------
        name_list : list of str
            Input star names to fetch IDs for.  Ignored when
            ``from_file`` is given.
        from_file : str, optional
            Path to a previously saved raw alternate-ID file.  If given,
            reads from disk (no network call); otherwise calls
            :meth:`download`.
        format : str, optional
            Astropy table format string (default ``'ascii'``).

        Returns
        -------
        ids : `~astropy.table.Table`
            Two-column table: ``input_col`` (input star name) and
            ``id_col`` (alternate identifier).  One row per
            (star, identifier) pair; null-sentinel rows are excluded by
            :meth:`preprocess`.
        """
        raw = self.load_raw(from_file, format=format) if from_file else self.download(name_list)
        return self.preprocess(raw)
