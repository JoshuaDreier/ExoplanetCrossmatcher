import pyvo
from astropy.table import Table
from crossmatching import Crossmatcher


if __name__ == "__main__":
    # Download NEA catalog
    nasa = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    pscomppars = nasa.run_sync("SELECT * FROM pscomppars").to_table()
    pscomppars.write("./pscomppars.txt", format='ascii', overwrite=True)
    print("Saved pscomppars to pscomppars.txt")

    # Download SIMBAD alternate IDs
    input_table = Table.read("./input/HPIC_LC4_combined_d50.txt", format="ascii")
    star_names = input_table["star_name"].tolist()

    cm = Crossmatcher()
    cm.load_alternate_ids(star_names)
    cm.alternate_ids.write("./alternate_ids.txt", format='ascii', overwrite=True)
    print(f"Saved {len(cm.alternate_ids)} alternate ID pairs to alternate_ids.txt")
