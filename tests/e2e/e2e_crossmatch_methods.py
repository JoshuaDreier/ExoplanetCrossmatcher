import os
from typing import NamedTuple
from crossmatching import Crossmatcher, NEACatalog, EMCCatalog, SimbadIdSupplier, EMCIdSupplier
import pytest

class _Config(NamedTuple):
    catalog_cls: type
    supplier_cls: type
    catalog_file: str
    catalog_format: str
    alt_ids_file: str
    expected_planets_col: str
    label: str


CONFIGS = [
    _Config(NEACatalog, SimbadIdSupplier, "tests/data/pscomppars_20260611.txt", "ascii", "tests/data/alternate_ids_hpic_20260611.txt", "nea_expected_planets", "nea-simbad"),
    _Config(NEACatalog, EMCIdSupplier,    "tests/data/pscomppars_20260611.txt", "ascii", "exo-mercat.csv",                              "nea_expected_planets", "nea-emc"),
    _Config(EMCCatalog, SimbadIdSupplier, "exo-mercat.csv",                     "csv",   "tests/data/alternate_ids_hpic_20260611.txt",  "emc_expected_planets", "emc-simbad"),
    _Config(EMCCatalog, EMCIdSupplier,    "exo-mercat.csv", "csv",   "exo-mercat.csv",           "emc_expected_planets", "emc-emc"),
]


def id_crossmatch(crossmatcher: Crossmatcher, query_row):
    crossmatcher.load_alternate_ids(query_row["star_name"].tolist(), from_file=crossmatcher._alt_ids_file)
    return crossmatcher.id_crossmatch(query_row, "star_name")

def coordinate_crossmatch(crossmatcher: Crossmatcher, query_row):
    return crossmatcher.coordinate_crossmatch(query_row, "star_name", input_epoch=2000)

def combined_crossmatch(crossmatcher: Crossmatcher, query_row):
    crossmatcher.load_alternate_ids(query_row["star_name"].tolist(), from_file=crossmatcher._alt_ids_file)
    return crossmatcher.combined_crossmatch(query_row, "star_name", input_epoch=2000)


@pytest.fixture(scope="session", params=CONFIGS, ids=[c.label for c in CONFIGS])
def loaded_matcher(request):
    config = request.param
    for f in (config.catalog_file, config.alt_ids_file):
        if not os.path.exists(f):
            pytest.fail(f"{f} not present")
    cm = Crossmatcher(config.catalog_cls(), config.supplier_cls())
    cm.load_catalog(from_file=config.catalog_file, format=config.catalog_format)
    cm._alt_ids_file = config.alt_ids_file
    cm._expected_planets_col = config.expected_planets_col
    return cm


@pytest.fixture(scope="function")
def stateless_matcher(loaded_matcher):
    cm = Crossmatcher(
        type(loaded_matcher.catalog)(),
        type(loaded_matcher.id_supplier)(),
    )
    cm._cache_catalog(loaded_matcher.catalog_table)
    cm._alt_ids_file = loaded_matcher._alt_ids_file
    cm._expected_planets_col = loaded_matcher._expected_planets_col
    yield cm
