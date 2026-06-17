import numpy as np
from astropy.table import Table
from crossmatching.enrichment.merger import ParamFiller, ParamQty, ParamStr

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
    
    params_q = {key: [ParamQty() for _ in range(len(t))] for key in filler.param_names_quantities}
    params_s = {key: [ParamStr() for _ in range(len(t))] for key in filler.param_names_strings}
    
    filler._merge_values(t, params_q, params_s, 'err1', 'err2')
    
    # Row 1 tests: rad should come from S1, teff from S2
    assert params_q['st_rad'][0].val == 1.5
    assert params_q['st_rad'][0].src == 'S1'
    assert params_q['st_rad'][0].err1 == 0.1
    
    assert params_q['st_teff'][0].val == 5000
    assert params_q['st_teff'][0].src == 'S2'
    assert params_q['st_teff'][0].err1 == 100
    
    # Row 2 tests: rad might be derived from teff!
    assert params_q['st_teff'][1].val == 6000
    assert params_q['st_teff'][1].src == 'S2'
    # rad should be derived, not missing, because teff=6000 is enough for ms_radius_from_teff
    assert params_q['st_rad'][1].mask == False
    assert 'ms(teff:S2)' in params_q['st_rad'][1].src or 'derived' in params_q['st_rad'][1].src

def test_merge_values_with_string_params():
    s1 = MockSource("S1", {1: {'spec': 'G2V'}})
    
    filler = ParamFiller([s1])
    t = Table({'id': [1, 2]})
    
    params_q = {key: [ParamQty() for _ in range(len(t))] for key in filler.param_names_quantities}
    params_s = {key: [ParamStr() for _ in range(len(t))] for key in filler.param_names_strings}
    
    filler._merge_values(t, params_q, params_s, 'err1', 'err2')
    
    assert params_s['st_spectype'][0].val == 'G2V'
    assert params_s['st_spectype'][0].src == 'S1'
    assert params_s['st_spectype'][0].mask == False
    
    assert params_s['st_spectype'][1].mask == True
