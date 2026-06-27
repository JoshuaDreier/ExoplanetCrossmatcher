import os, pathlib, glob, sys
import numpy as np
from astropy.table import Table

ROOT = pathlib.Path.cwd()
while not (ROOT / 'crossmatching').exists():
    ROOT = ROOT.parent
    print(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from crossmatching import Crossmatcher, EMCCatalog, EMCIdSupplier, ParamFiller
from crossmatching.enrichment import (
    HpicParamSource, NeaParamSource, SimbadParamSource,
    EpicParamSource, ToiParamSource, EuParamSource
)

input_table = Table.read('./input/HPIC_LC4_combined_d50.txt', format='ascii',)
cme = Crossmatcher(EMCCatalog(), EMCIdSupplier())

cme.load_catalog(from_file='./input/exo-mercat.csv', format='csv')
cme.load_alternate_ids(input_table['star_name'].tolist(), from_file='./input/exo-mercat.csv')

input_table = cme.remove_duplicates(input_table, input_starname_key='star_name')
out_emc = cme.combined_crossmatch(input_table, input_starname_key='star_name')


nea_src = NeaParamSource()
nea_src.load(from_file='././input/pscomppars.txt', format='ascii')

eu_src = EuParamSource()
eu_path = sorted(glob.glob('../Exo-MerCat/InputSources/eu_init*.csv'))[-1]
eu_src.load(from_file=eu_path, format="ascii.csv")

epic_src = EpicParamSource()
epic_path = sorted(glob.glob('../Exo-MerCat/InputSources/epic_init*.csv'))[-1]
epic_src.load(from_file=epic_path, format='ascii.csv')

toi_src = ToiParamSource()
toi_path = sorted(glob.glob('../Exo-MerCat/InputSources/toi_init*.csv'))[-1]
toi_src.load(from_file=toi_path, format='ascii.csv')

simbad_src = SimbadParamSource()
simbad_src.load(from_file='./input/simbad_params.txt')

hpic_src = HpicParamSource(out_emc)
hpic_src.load()


merger = ParamFiller([nea_src, eu_src, toi_src, toi_src, simbad_src, hpic_src])

e_emc = merger.enrich(
    cme.catalog_table,
    **EMCCatalog.ENRICH_KEYS,
    id_supplier=cme.id_supplier,
    alternate_ids=cme.alternate_ids
<<<<<<< HEAD
<<<<<<< HEAD
)[0]
=======
)
>>>>>>> a8d4a3a (added enrichment analysis notebook)
=======
)[0]
>>>>>>> 1c3f2a9 (anticipating new enrich behavior)

total = len(e_emc)

err_cols = [
    ('st_teff_max', 'Teff'), ('st_rad_max', 'Radius'), ('st_mass_max', 'Mass'),
    ('st_logg_max', 'logg'), ('st_met_max', 'Metallicity'),
    ('sy_vmag_max', 'Vmag'), ('sy_dist_max', 'Distance'),
]
print(f"{'Parameter':<14}  {'With error':>10}  {'%':>5}")
print('-' * 36)
for col, label in err_cols:
    n = (~np.ma.getmaskarray(e_emc[col])).sum()
    print(f'{label:<14}  {n:>10,}  {100*n/total:>4.1f}%')
    