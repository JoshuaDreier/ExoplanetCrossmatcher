from astropy.table import Table
from crossmatching import NEACatalog, SimbadIdSupplier


if __name__ == "__main__":
    # Download and cache the NEA catalog (raw, before preprocessing)
    NEACatalog().save_raw("pscomppars.txt")
    print("Saved pscomppars to pscomppars.txt")

    # Download raw SIMBAD alternate IDs for all HPIC input stars (before preprocessing)
    input_table = Table.read("./input/HPIC_LC4_combined_d50.txt", format="ascii")
    star_names = input_table["star_name"].tolist()
    SimbadIdSupplier().save_raw(star_names, "./alternate_ids.txt")
    print("Saved raw alternate IDs to alternate_ids.txt")
