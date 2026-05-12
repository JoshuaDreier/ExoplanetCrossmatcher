from crossmatching import Crossmatcher
from astropy.table import Table
import pytest

def id_crossmatch(crossmatcher: Crossmatcher, query_row):
    # print("$$$", type(query_row))
    # print("$$$", query_row["star_name"])
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
    cm = loaded_matcher
    yield cm
    cm.alternate_ids = Table()
    cm.alternate_ids_cached = False
    cm.id_matched = Table()
    cm.coords3d_matched = Table()
    cm.coords2d_matched = Table()
