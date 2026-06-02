from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier
import pytest

def id_crossmatch(crossmatcher: Crossmatcher, query_row):
    crossmatcher.load_alternate_ids(query_row["star_name"].tolist(), from_file="alternate_ids.txt")
    return crossmatcher.id_crossmatch(query_row, "star_name")

def coordinate_crossmatch(crossmatcher: Crossmatcher, query_row):
    return crossmatcher.coordinate_crossmatch(query_row, "star_name")

def combined_crossmatch(crossmatcher: Crossmatcher, query_row):
    crossmatcher.load_alternate_ids(query_row["star_name"].tolist(), from_file="alternate_ids.txt")
    return crossmatcher.combined_crossmatch(query_row, "star_name")

@pytest.fixture(scope="session")
def loaded_matcher():
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm.load_catalog(from_file="pscomppars.txt")
    return cm


@pytest.fixture(scope="function")
def stateless_matcher(loaded_matcher):
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm._cache_catalog(loaded_matcher.catalog_table)  # reference, not a reload
    yield cm
