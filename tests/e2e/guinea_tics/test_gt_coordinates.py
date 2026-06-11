import pytest
from tests.e2e.e2e_utils import _e2e_all_planets_found, _e2e_no_false_positives
from tests.e2e.guinea_tics.gt_utils import get_guinea_tics_table_parametrization
from tests.e2e.e2e_crossmatch_methods import coordinate_crossmatch

gt_ids, gt_row_params = get_guinea_tics_table_parametrization()

@pytest.mark.xfail(strict=False, reason="some stars only reachable via combined method")
@pytest.mark.parametrize(
    "star_name,ra,dec,planets_by_catalog",
    gt_row_params,
    ids=[f"2d-af-xfail-{gt_id}" for gt_id in gt_ids],
)
def test_e2e_guinea_tics_coordinates_2d_all_found(stateless_matcher, star_name, ra, dec, planets_by_catalog):
    planets = planets_by_catalog[stateless_matcher._expected_planets_col]
    _e2e_all_planets_found(star_name, ra, dec, planets, stateless_matcher, coordinate_crossmatch)

@pytest.mark.xfail(strict=False, reason="some stars only reachable via combined method")
@pytest.mark.parametrize(
    "star_name,ra,dec,planets_by_catalog",
    gt_row_params,
    ids=[f"2d-fp-xfail-{gt_id}" for gt_id in gt_ids],
)
def test_e2e_guinea_tics_coordinates_2d_no_false_positives(stateless_matcher, star_name, ra, dec, planets_by_catalog):
    planets = planets_by_catalog[stateless_matcher._expected_planets_col]
    _e2e_no_false_positives(star_name, ra, dec, planets, stateless_matcher, coordinate_crossmatch)
