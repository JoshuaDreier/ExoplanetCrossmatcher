from astropy.table import Table
from pytest_check import check


def get_planets_for_star(star_name, ra, dec, sy_dist, crossmatcher, crossmatching_method):
    """Helper function to get planets for a star using the specified crossmatching method"""
    query_row = Table(
        rows=[(star_name, ra, dec, sy_dist)],
        names=["star_name", "ra", "dec", "sy_dist"],
    )
    results = crossmatching_method(crossmatcher, query_row)
    return results["pl_name"].tolist()


def _e2e_all_planets_found(star_name, ra, dec, sy_dist, expected_planets, crossmatcher, crossmatching_method):
    """End-to-end test for a single star, seeing if all planets were found"""
    matched_planets = get_planets_for_star(star_name, ra, dec, sy_dist, crossmatcher, crossmatching_method)
    
    assert len(matched_planets) != 0, f"No planets found for {star_name} with {crossmatching_method.__name__}, expected {len(expected_planets)}"
    for planet in expected_planets:
        check.is_in(planet, matched_planets, f"Expected planet '{planet}' not found for '{star_name}'")

    print(f"{star_name}\n{expected_planets = }\n{matched_planets = }")

def _e2e_no_false_positives(star_name, ra, dec, sy_dist, expected_planets, crossmatcher, crossmatching_method):
    """End-to-end test for a single star, seeing that there are no false positives"""
    matched_planets = get_planets_for_star(star_name, ra, dec, sy_dist, crossmatcher, crossmatching_method)

    for planet in matched_planets:
        check.is_in(planet, expected_planets, f"Unexpected planet '{planet}' found for '{star_name}'")

    print(f"{star_name}\n{expected_planets = }\n{matched_planets = }")


