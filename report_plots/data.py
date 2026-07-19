"""Shared data-building steps for the paper plot notebooks.

Wraps the crossmatching + enrichment pipeline. All input paths are
resolved relative to the repo root, so the functions work regardless of the
kernel's working directory.
"""

import numpy as np
from astropy.table import Table, Column
from astropy import units as u

from crossmatching import (
    Crossmatcher, EMCCatalog, EMCIdSupplier, NEACatalog, SimbadIdSupplier,
    ParamFiller, rocky_mask, temperate_mask,
)
from crossmatching.enrichment import (
    NeaParamSource, SimbadParamSource, EpicParamSource, ToiParamSource, EuParamSource,
)

from .common import ROOT, HZ_INNER, HZ_OUTER

INPUT_TABLE_PATH = ROOT / "input" / "HPIC_LC4_combined_d50.txt"
EMC_CSV_PATH = ROOT / "input" / "exo-mercat.csv"
PSCOMPPARS_PATH = ROOT / "input" / "pscomppars.txt"
SIMBAD_PARAMS_PATH = ROOT / "input" / "simbad_params.txt"
ALTERNATE_IDS_PATH = ROOT / "input" / "alternate_ids_hpic.txt"
# EU/EPIC/TOI param sources, snapshotted into input/ from a local Exo-MerCat run
# (eu_init2026-07-11, epic_init2026-06-15, toi_init2026-07-11)
EU_PATH = ROOT / "input" / "eu_init.csv"
EPIC_PATH = ROOT / "input" / "epic_init.csv"
TOI_PATH = ROOT / "input" / "toi_init.csv"


def load_input_table():
    """HPIC LC4 input stellar survey (~15000 stars within 50 pc)."""
    return Table.read(str(INPUT_TABLE_PATH), format="ascii")


def emc_crossmatcher(**crossmatcher_kwargs):
    """EMC crossmatcher with catalog and alternate IDs loaded (no crossmatch yet)."""
    input_table = load_input_table()
    cme = Crossmatcher(EMCCatalog(), EMCIdSupplier(), **crossmatcher_kwargs)
    cme.load_catalog(from_file=str(EMC_CSV_PATH), format="csv")
    cme.load_alternate_ids(input_table["star_name"].tolist(), from_file=str(EMC_CSV_PATH))
    return cme, input_table


def emc_crossmatch(dedup_input=True, **crossmatcher_kwargs):
    """Run the combined HPIC x Exo-MerCat crossmatch; returns (cme, input_table, matched).

    dedup_input=True removes duplicate stars from the input table first (as the
    population-analysis pipeline does); the yield notebook keeps the raw input.
    """
    cme, input_table = emc_crossmatcher(**crossmatcher_kwargs)
    if dedup_input:
        input_table = cme.remove_duplicates(input_table, input_starname_key="star_name")
    matched = cme.combined_crossmatch(input_table, input_starname_key="star_name")
    return cme, input_table, matched


def nea_crossmatch(**crossmatcher_kwargs):
    """Run the combined HPIC x NEA crossmatch; returns (cm, input_table, matched)."""
    input_table = load_input_table()
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier(), **crossmatcher_kwargs)
    cm.load_catalog(from_file=str(PSCOMPPARS_PATH))
    cm.load_alternate_ids(input_table["star_name"], from_file=str(ALTERNATE_IDS_PATH))
    matched = cm.combined_crossmatch(input_table, input_starname_key="star_name")
    return cm, input_table, matched


def build_param_filler():
    """Priority-ordered ParamFiller: NEA -> EU -> EPIC -> TOI -> SIMBAD.

    (Input-table values always keep highest priority inside enrich() itself.)
    """
    nea_src = NeaParamSource()
    nea_src.load(from_file=str(PSCOMPPARS_PATH), format="ascii")

    eu_src = EuParamSource()
    eu_src.load(from_file=str(EU_PATH), format="ascii.csv")

    epic_src = EpicParamSource()
    epic_src.load(from_file=str(EPIC_PATH), format="ascii.csv")

    toi_src = ToiParamSource()
    toi_src.load(from_file=str(TOI_PATH), format="ascii.csv")

    simbad_src = SimbadParamSource()
    simbad_src.load(from_file=str(SIMBAD_PARAMS_PATH))

    return ParamFiller([nea_src, eu_src, epic_src, toi_src, simbad_src])


def enrich(table, cme, merger, disable_calculations=False):
    """Enrich an EMC-schema table (the crossmatch output or cme.catalog_table)."""
    return merger.enrich(
        table,
        **EMCCatalog.ENRICH_KEYS,
        id_supplier=cme.id_supplier,
        alternate_ids=cme.alternate_ids,
        disable_calculations=disable_calculations,
    )[0]


def compute_is_rocky_temperate(catalog):
    """Add r_earth/mass_earth columns and the three-way rocky/temperate status.

    'Confirmed' requires central radius+insolation inside the box AND EMC
    status CONFIRMED; 'Uncertain' is interval overlap in both quantities,
    regardless of confirmation status.
    """
    catalog['mass_earth'] = catalog['mass'] * u.M_jupiter.to(u.M_earth)
    catalog['msini_earth'] = catalog['msini'] * u.M_jupiter.to(u.M_earth)
    catalog['r_earth'] = catalog['r'] * u.R_jupiter.to(u.R_earth)
    catalog['r_earth_max'] = catalog['r_max'] * u.R_jupiter.to(u.R_earth)
    catalog['r_earth_min'] = catalog['r_min'] * u.R_jupiter.to(u.R_earth)
    catalog['r_earth_lower_bound'] = catalog['r_lower_bound'] * u.R_jupiter.to(u.R_earth)
    catalog['r_earth_upper_bound'] = catalog['r_upper_bound'] * u.R_jupiter.to(u.R_earth)

    rocky_confirmed = rocky_mask(
        catalog['r_earth'], catalog['r_earth_min'], catalog['r_earth_max'], catalog['r_earth_lower_bound'], catalog['r_earth_upper_bound'],
        use_interval=False
    )
    rocky_uncertain = rocky_mask(
        catalog['r_earth'], catalog['r_earth_min'], catalog['r_earth_max'], catalog['r_earth_lower_bound'], catalog['r_earth_upper_bound'],
        use_interval=True
    )
    catalog['is_rocky'] = Column([''] * len(catalog), dtype="U10")
    catalog['is_rocky'][rocky_uncertain] = ['Uncertain'] * sum(rocky_uncertain)
    catalog['is_rocky'][rocky_confirmed] = ['Confirmed'] * sum(rocky_confirmed)

    temperate_confirmed = temperate_mask(
        catalog['pl_insol'], catalog['pl_insol_max'], catalog['pl_insol_min'],
        HZ_OUTER, HZ_INNER,
        use_interval=False
    )
    temperate_uncertain = temperate_mask(
        catalog['pl_insol'], catalog['pl_insol_max'], catalog['pl_insol_min'],
        HZ_OUTER, HZ_INNER,
        use_interval=True
    )
    catalog['is_temperate'] = Column([''] * len(catalog), dtype="U10")
    catalog['is_temperate'][temperate_uncertain] = ['Uncertain'] * sum(temperate_uncertain)
    catalog['is_temperate'][temperate_confirmed] = ['Confirmed'] * sum(temperate_confirmed)

    emc_confirmed = catalog["status"] == "CONFIRMED"
    rocky_temp_uncertain = temperate_uncertain & rocky_uncertain
    rocky_temp_confirmed = temperate_confirmed & rocky_confirmed & emc_confirmed
    catalog['rocky_temp_status'] = Column([''] * len(catalog), dtype="U10")
    catalog['rocky_temp_status'][rocky_temp_uncertain] = ["Uncertain"] * sum(rocky_temp_uncertain)
    catalog['rocky_temp_status'][rocky_temp_confirmed] = ["Confirmed"] * sum(rocky_temp_confirmed)


def rocky_temperate_summary(**catalogs):
    """Print the Confirmed/Uncertain rocky+temperate counts per named catalog."""
    for name, catalog in catalogs.items():
        n_conf = sum(catalog['rocky_temp_status'] == 'Confirmed')
        n_unc = sum(catalog['rocky_temp_status'] == 'Uncertain')
        print(f"{name}: {n_conf} Confirmed + {n_unc} Uncertain rocky, temperate planets")
