import pytest
from pathlib import Path
from astropy.table import Table
from crossmatching import Crossmatcher
from tests.e2e.e2e_utils import test_e2e_all_planets_found, test_e2e_no_false_positives
from tests.e2e.guinea_tics.gt_utils import get_guinea_tics_table_parametrization
from tests.e2e.e2e_crossmatch_methods import coordinate_crossmatch

gt_ids, gt_row_params = get_guinea_tics_table_parametrization()

@pytest.mark.parametrize(
    "star_name,ra,dec,sy_dist,expected_planets",
    gt_row_params,
    ids=gt_ids,
)
def test_e2e_guinea_tics_id_all_found(star_name, ra, dec, sy_dist, expected_planets):
    test_e2e_all_planets_found(star_name, ra, dec, sy_dist, expected_planets, coordinate_crossmatch)    

@pytest.mark.parametrize(
    "star_name,ra,dec,sy_dist,expected_planets",
    gt_row_params,
    ids=gt_ids,
)
def test_e2e_guinea_tics_id_no_false_positives(star_name, ra, dec, sy_dist, expected_planets):
    test_e2e_no_false_positives(star_name, ra, dec, sy_dist, expected_planets, coordinate_crossmatch)    
