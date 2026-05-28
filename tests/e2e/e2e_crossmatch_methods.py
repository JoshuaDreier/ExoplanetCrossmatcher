from crossmatching import Crossmatcher
import pytest

def id_crossmatch(crossmatcher: Crossmatcher, query_row):
    crossmatcher.load_alternate_ids(query_row["star_name"].tolist(), from_file="alternate_ids.txt")
    return crossmatcher.id_crossmatch(query_row)

def coordinate_crossmatch_3d(crossmatcher: Crossmatcher, query_row):
    return crossmatcher.coordinate_crossmatch(query_row)[0]

def coordinate_crossmatch_2d(crossmatcher: Crossmatcher, query_row):
    return crossmatcher.coordinate_crossmatch(query_row)[1]

def combined_crossmatch(crossmatcher: Crossmatcher, query_row):
    crossmatcher.load_alternate_ids(query_row["star_name"].tolist(), from_file="alternate_ids.txt")
    return crossmatcher.combined_crossmatch(query_row)

@pytest.fixture(scope="session")
def loaded_matcher():
    cm = Crossmatcher()
    cm.load_catalog(from_file="pscomppars.txt")
    return cm


@pytest.fixture(scope="function")
def stateless_matcher(loaded_matcher):
    cm = Crossmatcher()
    cm.catalogue = loaded_matcher.catalogue  # reference, not a reload
    cm.catalogue_cached = True
    yield cm
