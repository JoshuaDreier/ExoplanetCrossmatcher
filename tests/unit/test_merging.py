import numpy as np
from astropy.table import Table
from crossmatching.enrichment.merger import ParamFiller, _ParamQtyArrays, _ParamStrArrays

class MockSource:
    def __init__(self, name, data):
        self.source_name = name
        self.data = data
        
    def get(self, row, **kwargs):
        return self.data.get(row['id'], {})

def test_merge_values_priority():
    s1 = MockSource("S1", {1: {'rad': 1.5, 'rad_err1': 0.1}})
    s2 = MockSource("S2", {1: {'rad': 2.0, 'teff': 5000, 'teff_err1': 100}, 2: {'teff': 6000}})
    
    filler = ParamFiller([s1, s2])
    
    t = Table({'id': [1, 2]})
    
    params_q = {key: _ParamQtyArrays(len(t)) for key in filler.param_names_quantities}
    params_s = {key: _ParamStrArrays(len(t)) for key in filler.param_names_strings}
    
    filler._merge_values(t, params_q, params_s, 'err1', 'err2')
    
    # Row 1 tests: rad should come from S1, teff from S2
    assert params_q['st_rad'].val[0] == 1.5
    assert params_q['st_rad'].src[0] == 'S1'
    assert params_q['st_rad'].err1[0] == 0.1
    
    assert params_q['st_teff'].val[0] == 5000
    assert params_q['st_teff'].src[0] == 'S2'
    assert params_q['st_teff'].err1[0] == 100
    
    # Row 2 tests: rad might be derived from teff!
    assert params_q['st_teff'].val[1] == 6000
    assert params_q['st_teff'].src[1] == 'S2'
    # rad should be derived, not missing, because teff=6000 is enough for ms_radius_from_teff
    assert params_q['st_rad'].mask[1] == False
    assert 'ms(teff:S2)' in params_q['st_rad'].src[1] or 'derived' in params_q['st_rad'].src[1]

def test_merge_values_with_string_params():
    s1 = MockSource("S1", {1: {'spec': 'G2V'}})
    
    filler = ParamFiller([s1])
    t = Table({'id': [1, 2]})
    
    params_q = {key: _ParamQtyArrays(len(t)) for key in filler.param_names_quantities}
    params_s = {key: _ParamStrArrays(len(t)) for key in filler.param_names_strings}
    
    filler._merge_values(t, params_q, params_s, 'err1', 'err2')
    
    assert params_s['st_spectype'].val[0] == 'G2V'
    assert params_s['st_spectype'].src[0] == 'S1'
    assert params_s['st_spectype'].mask[0] == False
    
    assert params_s['st_spectype'].mask[1] == True
