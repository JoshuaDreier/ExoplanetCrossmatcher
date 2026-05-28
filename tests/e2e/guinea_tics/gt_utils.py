from astropy.table import Table
from matplotlib import table

def get_guinea_tics_table_parametrization():
    table = Table.read("tests/data/guinea_tics.csv", format="csv")
    gt_row_params = []
    for row in table:
        expected_planets = [p for p in row["expected_planets"].split("|")]
        gt_row_params.append((row["star_name"], row["ra"], row["dec"], row["sy_dist"], expected_planets))

    gt_ids = table["hostname"].tolist()
    return gt_ids, gt_row_params
