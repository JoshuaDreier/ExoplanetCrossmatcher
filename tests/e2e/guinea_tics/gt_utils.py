from astropy.table import Table

def get_guinea_tics_table_parametrization():
    table = Table.read("tests/data/guinea_tics.csv", format="csv")
    expected_cols = [c for c in table.colnames if c.endswith("_expected_planets")]
    gt_row_params = []
    for row in table:
        planets_by_catalog = {
            col: [p for p in str(row[col]).split("|")] if str(row[col]) else []
            for col in expected_cols
        }
        gt_row_params.append((row["star_name"], row["ra"], row["dec"], planets_by_catalog))

    gt_ids = table["hostname"].tolist()
    return gt_ids, gt_row_params
