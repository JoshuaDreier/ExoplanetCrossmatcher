import pytest
from tests.e2e.e2e_utils import _e2e_all_planets_found, _e2e_no_false_positives
from tests.e2e.guinea_tics.gt_utils import get_guinea_tics_table_parametrization
from tests.e2e.e2e_crossmatch_methods import combined_crossmatch, stateless_matcher, loaded_matcher

gt_ids, gt_row_params = get_guinea_tics_table_parametrization()

@pytest.mark.parametrize(
    "star_name,ra,dec,sy_dist,expected_planets",
    gt_row_params,
    ids=[f"combined-af-{gt_id}" for gt_id in gt_ids],
)
def test_e2e_guinea_tics_combined_all_found(stateless_matcher, star_name, ra, dec, sy_dist, expected_planets):
    _e2e_all_planets_found(star_name, ra, dec, sy_dist, expected_planets, stateless_matcher, combined_crossmatch)    


@pytest.mark.parametrize(
    "star_name,ra,dec,sy_dist,expected_planets",
    gt_row_params,
    ids=[f"combined-fp-{gt_id}" for gt_id in gt_ids],
)
def test_e2e_guinea_tics_combined_no_false_positives(stateless_matcher, star_name, ra, dec, sy_dist, expected_planets):
    _e2e_no_false_positives(star_name, ra, dec, sy_dist, expected_planets, stateless_matcher, combined_crossmatch)    
