from crossmatching import Crossmatcher
from astropy.table import Table

def id_crossmatch(query_row):
    cm = Crossmatcher()
    cm.load_catalog(from_file="pscomppars.txt") # "defacto "mocking", TODO: make independent of web
    return cm.id_crossmatch(query_row)

def coordinate_crossmatch(query_row):
    cm = Crossmatcher()
    cm.load_catalog(from_file="pscomppars.txt")
    return cm.coordinate_crossmatch(query_row)

def combined_crossmatch(query_row):
    cm = Crossmatcher()
    cm.load_catalog(from_file="pscomppars.txt")
    return cm.combined_crossmatch(query_row)