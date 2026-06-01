import pytest
from astropy.table import Table, MaskedColumn
from crossmatching import Crossmatcher, NEACatalog, SimbadIdSupplier


@pytest.fixture
def cm_with_duplicates():
    """
        Star A and Star B share 'shared_id'
        Star C is unique
    """
    cm = Crossmatcher(NEACatalog(), SimbadIdSupplier())
    cm.alternate_ids = Table({
        "input_ids": ["Star A", "Star B", "Star C"],
        "id": MaskedColumn(["shared_id", "shared_id", "unique_id"], mask=[False, False, False]),
    })
    cm.alternate_ids_cached = True
    return cm


@pytest.fixture
def input_table():
    return Table({"star_name": ["Star A", "Star B", "Star C"]})
