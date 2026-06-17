import os
import pytest
from astropy.table import Table

from crossmatching import Crossmatcher, EMCCatalog, SimbadIdSupplier, EMCIdSupplier

_EMC_FILE   = "tests/data/exo-mercat2026-06-08.csv"
_HPIC_FILE  = "tests/data/HPIC_LC4_combined_d50_20260611.txt"
_ALT_FILE   = "tests/data/alternate_ids_hpic_20260611.txt"
_UUID       = "exo-mercat_name"

_files_present = all(os.path.exists(f) for f in (_EMC_FILE, _HPIC_FILE, _ALT_FILE))


@pytest.mark.skipif(not _files_present, reason="data files not present")
def test_emc_id_supplier_finds_superset_of_simbad():
    """
    After the whitespace normalization fix, EMCIdSupplier must find every planet
    that SimbadIdSupplier finds (plus more). This test fails before the fix because
    raw SIMBAD multi-space alias forms (e.g. "Ross  508") stored in main_id_aliases
    don't match the single-spaced HPIC input names without normalization.
    """
    input_table = Table.read(_HPIC_FILE, format="ascii")
    name_list   = input_table["star_name"].tolist()

    cm_simbad = Crossmatcher(EMCCatalog(), SimbadIdSupplier())
    cm_simbad.load_catalog(from_file=_EMC_FILE, format="csv")
    cm_simbad.load_alternate_ids(name_list, from_file=_ALT_FILE)
    simbad_planets = set(
        cm_simbad.id_crossmatch(input_table, input_starname_key="star_name")[_UUID].tolist()
    )

    cm_emc = Crossmatcher(EMCCatalog(), EMCIdSupplier())
    cm_emc.load_catalog(from_file=_EMC_FILE, format="csv")
    cm_emc.load_alternate_ids(name_list, from_file=_EMC_FILE)
    emc_planets = set(
        cm_emc.id_crossmatch(input_table, input_starname_key="star_name")[_UUID].tolist()
    )

    missing = simbad_planets - emc_planets
    assert not missing, (
        f"EMCIdSupplier is missing {len(missing)} planet(s) found by SimbadIdSupplier:\n"
        + "\n".join(sorted(missing))
    )
