from astropy.table import Table
from crossmatching.enrichment import ParamFiller
from tests.enrich_keys import DEFAULT_ENRICH_KEYS

def test_enrich_returns_resolved_cols():
    """Verify that ParamFiller.enrich() returns a list of resolved columns as the second output."""
    cat = Table({
        "nasa_name": ["Planet X"],
    })
    
    table, resolved_cols = ParamFiller([]).enrich(cat, **DEFAULT_ENRICH_KEYS, disable_calculations=True)
    
    expected_cols = [
        DEFAULT_ENRICH_KEYS['planet_radius_key'],
        DEFAULT_ENRICH_KEYS['planet_mass_key'],
        DEFAULT_ENRICH_KEYS['planet_flux_key'],
        DEFAULT_ENRICH_KEYS['planet_equilibrium_temperature_key'],
        DEFAULT_ENRICH_KEYS['semi_major_axis_key'],
        DEFAULT_ENRICH_KEYS['period_key'],
        DEFAULT_ENRICH_KEYS['msini_key'],
        DEFAULT_ENRICH_KEYS['star_radius_key'],
        DEFAULT_ENRICH_KEYS['star_mass_key'],
        DEFAULT_ENRICH_KEYS['star_effective_temperature_key'],
        DEFAULT_ENRICH_KEYS['star_logg_key'],
        DEFAULT_ENRICH_KEYS['star_metallicity_key'],
        DEFAULT_ENRICH_KEYS['star_luminosity_key'],
        DEFAULT_ENRICH_KEYS['vmag_key'],
        DEFAULT_ENRICH_KEYS['kmag_key'],
        DEFAULT_ENRICH_KEYS['distance_key'],
        DEFAULT_ENRICH_KEYS['star_spectral_type_key'],
    ]
    
    assert isinstance(resolved_cols, list)
    for col in expected_cols:
        assert col in resolved_cols
