# Architecture

## Package layout

```
crossmatching/
  crossmatcher.py       # Crossmatcher class + allowed_angular_separation()
  config.py             # crossmatching.cfg defaults
  __init__.py           # Public re-exports
  catalogs/
    base.py             # CatalogBase — abstract interface
    nea.py              # NEACatalog — NASA Exoplanet Archive (pscomppars)
    exomercat.py        # EMCCatalog — Exo-MerCat merged catalog
    file.py             # FileCatalog — load from arbitrary local file
  id_suppliers/
    base.py             # IdSupplierBase — abstract interface
    simbad.py           # SimbadIdSupplier — SIMBAD TAP
    emc.py              # EMCIdSupplier — reads aliases from EMC file
  enrichment/
    merger.py           # ParamFiller — priority-ordered parameter merging
    inference.py        # infer_* functions for derived parameters
    masks.py            # rocky_mask(), temperate_mask()
    spectral_types.py   # Spectral-type utilities
    radius_estimation.py# ms_radius_from_teff, mass_radius_chen_kipping
    __init__.py         # Public re-exports
    param_sources/
      base.py           # ParamSource — abstract interface + build helpers
      hpic.py           # HpicParamSource — HPIC crossmatch output
      nea.py            # NeaParamSource — NEA pscomppars
      simbad.py         # SimbadParamSource — SIMBAD mesFe_h tables
      epic.py           # EpicParamSource — K2 EPIC catalog
      toi.py            # ToiParamSource — TESS TOI catalog
      eu.py             # EuParamSource — exoplanet.eu catalog
```


## Enrichment pipeline
```mermaid
%%{init: {"theme": "base", "layout": "elk", "elk": {"nodePlacementStrategy": "BRANDES_KOEPF"}}}%%
flowchart TB
    subgraph CAT["EMC Catalog"]
            direction LR
            R_J["$$r_p$$"]
            MSI["$$m\sin i$$"]
            A_C["$$a$$"]
            P_C["$$P$$"]
        end
    subgraph SRCS["Stellar Param Sources"]
        direction TB
        subgraph CATS["Input Catalogs"]
            direction TB
            NEA["<b>NASA Exoplanet Archive</b>"]
            EPIC["<b>EPIC (K2)</b>"]
            TOI["<b>TOI (TESS)</b>"]
            SIM["<b>SIMBAD</b>"]
        end
        NEA  -->|"5946"| KMAG["K mag"]
        EPIC -->|"970"| KMAG
        SIM  -->|"8154"| KMAG
        NEA  -->|"2282"| SPEC["spec type"]
        EPIC -->|"141"| SPEC
        SIM  -->|"2905"| SPEC
        TEFF -->|"10769"| SPEC
        NEA  -->|"5941"| TEFF["$$T_{\mathrm{eff}}$$"]
        EPIC -->|"544"| TEFF
        TOI  -->|"5332"| TEFF
        SIM  -->|"2528"| TEFF
        NEA  -->|"5915"| LOGG["$$\log g$$"]
        EPIC -->|"430"| LOGG
        TOI  -->|"4819"| LOGG
        SIM  -->|"2559"| LOGG
        NEA  -->|"5612"| MET["$$[\text{Fe/H}]$$"]
        EPIC -->|"249"| MET
        TOI  -->|"0"| MET
        SIM  -->|"3497"| MET
        NEA  -->|"5919"| RSTAR["$$R_\star$$"]
        EPIC -->|"960"| RSTAR
        TOI  -->|"5059"| RSTAR
        TEFF --> |"348"| ZAMS["$$R_\star\approx(T_{\mathrm{eff}}/T_\odot)^{n(spec)}$$"]
        SPEC -->|"105"| ZAMS
        ZAMS -. "348" .-> RSTAR
        KMAG -->|"199"| MANN["$$R_\star=f(M_{K_s},[\text{Fe/H}])$$"]
        MET  -->|"148"| MANN
        MANN -. "199" .-> RSTAR
        TEFF -->|"118"| MANNTEFF["$$R_\star=f(T_{\text{eff}},[\text{Fe/H}])$$"]
        MET  -->|"21"| MANNTEFF
        MANNTEFF -. "118" .-> RSTAR
        TEFF -->|"1890"| TORRES["$$R_\star=f(T_{\text{eff}},\log g,[\text{Fe/H}])$$"]
        LOGG -->|"1890"| TORRES
        MET  -->|"1799"| TORRES
        TORRES -. "1890" .-> RSTAR
        NEA  -->|"6213"| MSTAR["$$M_\star$$"]
        EPIC -->|"288"| MSTAR
        LOGG  --> |"7510"| MLOG["$$M_\star=R_\star^2\cdot 10^{\log g-\log g_\odot}$$"]
        RSTAR --> |"7510"| MLOG
        MLOG  -. "7510" .-> MSTAR
        NEA   -->|"5924"| LSTAR["$$L_\star$$"]
        EPIC  -->|"111"| LSTAR
        RSTAR -->|"8311"| LUM["$$L_\star=R_\star^2(T_{\mathrm{eff}}/T_\odot)^4$$"]
        TEFF  --> |"8311"| LUM
        LUM   -. "8311" .-> LSTAR
    end

    FCALC["$$F=L_\star/a^2$$"]
    MSTAR -->|"5945"| KEPL["$$a = \left(M_\star \left(\frac{P}{{yr}}\right)^2\right)^{1/3}$$"]
    P_C   -->|"14784"| KEPL

    A_C  -->|"8644"| AFIN["$$a$$"]
    KEPL -. "5945" .-> AFIN
    AFIN  -->|"8644"| FCALC
    AFIN  -. "5945" .-> FCALC
    LSTAR  -->|"6035"| FCALC
    LSTAR  -. "8311" .-> FCALC

    subgraph CLASS["Classification"]
        direction LR
        ROCKY[/"rocky$$\; r_p\in[0.5,1.5]\,R_\oplus$$"/]
        TEMP[/"temperate$$\; F_{\mathrm{rel}}\in[0.35,1.7]\,S_\oplus$$"/]
    end

    NEA   -->|"4405"| FLUX["$$F_{\mathrm{rel}}$$"]
    EPIC  -->|"30"| FLUX
    TOI   -->|"5355"| FLUX
    FCALC -. "4361" .-> FLUX
    FLUX  -->|"9790"| TEMP
    FLUX  -. "4361" .-> TEMP

    R_J -->|"12744"| REARTH["$$r_p$$"]
    MSI -->|"1466"| CK["M-R-rel + msini + i prior [1,2]"]
    CK  -. "1466" .-> REARTH
    REARTH  -->|"1888 confirmed"| ROCKY
    REARTH  -. "70 uncertain" .-> ROCKY
    style NEA  fill:#dde8f7,stroke:#1a6fba,color:#1a6fba
    style EPIC fill:#fde9dc,stroke:#d4550d,color:#d4550d
    style TOI  fill:#dcf5dc,stroke:#2a8a2a,color:#2a8a2a
    style SIM  fill:#f0e0fa,stroke:#7b42a8,color:#7b42a8
    linkStyle 0,3,7,11,15,19,35,40,53 stroke:#1a6fba,stroke-width:2px
    linkStyle 1,4,8,12,16,20,36,41,54 stroke:#d4550d,stroke-width:2px
    linkStyle 9,13,17,21,55 stroke:#2a8a2a,stroke-width:2px
    linkStyle 2,5,10,14,18 stroke:#7b42a8,stroke-width:2px
```

**References**
[1] Chen & Kipping 2017, *ApJ* **834**, 17
[2] Stevens & Gaudi 2013, *PASP* **125**, 933
[3] Kopparapu et al. 2013, *ApJ* **765**, 131


The enrichment step adds stellar and derived planetary parameters to a crossmatch result.  Parameters are pulled from up to five sources in priority order the first source that provides a non-null value for a given parameter wins. This is quite crude and leads to likely inconsistent parameters, in the context of this package it was implemented to be able to compare its results to other papers

