import pytest
from astropy.table import Table

from astropy.table import Table

from crossmatching.enrichment.param_sources.base import ParamSource
from crossmatching.id_suppliers.base import IdSupplierBase


class _DummyParamSource(ParamSource):
    key_col = 'key'
    source_name = 'dummy'
    param_columns = {
        'teff': 'st_teff',
        'spec': 'st_spectype',
    }
    param_error_columns = {
        'teff': ('st_tefferr1', 'st_tefferr2'),
    }

    def download(self, key_list: list[str]) -> Table:
        raise NotImplementedError

    def _build_lookup(self, table: Table) -> dict:
        return {}


def test_param_source_get_falls_back_on_input_starname_via_id_supplier():
    class _StubIdSupplier(IdSupplierBase):
        def download(self, name_list: list[str]) -> Table:
            raise NotImplementedError

    src = _DummyParamSource()
    src._lookup = {
        'input-alias': {'teff': 4450.0},
    }
    row = {
        'key': '',
        'star_name': 'source-star',
    }
    alternate_ids = Table({
        'input_ids': ['source-star'],
        'id': ['input-alias'],
    })

    value = src.get(
        row,
        input_starname_key='star_name',
        id_supplier=_StubIdSupplier(),
        alternate_ids=alternate_ids,
    )

    assert value == {'teff': 4450.0}


def test_param_source_uses_param_columns_for_table_rows():
    src = _DummyParamSource()
    table = Table({
        'key': ['A'],
        'st_teff': [5800.0],
        'st_tefferr1': [120.0],
        'st_tefferr2': [-90.0],
        'st_spectype': ['G2V'],
    })
    row = table[0]

    assert src._get_float(row, 'teff') == pytest.approx(5800.0)
    assert src._get_str(row, 'spec') == 'G2V'
    assert src._get_err_pair(row, 'teff') == (120.0, 90.0)


def test_param_source_uses_param_columns_for_dict_rows():
    src = _DummyParamSource()
    row = {
        'st_teff': 5800.0,
        'st_tefferr1': 120.0,
        'st_tefferr2': -90.0,
        'st_spectype': 'G2V',
    }

    assert src._get_float(row, 'teff') == pytest.approx(5800.0)
    assert src._get_str(row, 'spec') == 'G2V'
    assert src._get_err_pair(row, 'teff') == (120.0, 90.0)
