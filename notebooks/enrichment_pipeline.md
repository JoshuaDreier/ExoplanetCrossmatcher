# Enrichment Pipeline

```mermaid
%%{init: {"theme": "base", "flowchart": {"htmlLabels": true}}}%%
flowchart TD

    %% ── ① Stellar Parameter Sources
    subgraph SRCS["① Stellar Param Sources — first-source-wins priority merge"]
        direction LR
        HPIC["<b>HPIC-LC4</b><br/>$$T_{\mathrm{eff}}\cdot R_\star\cdot M_\star$$<br/>$$v_{\mathrm{mag}}\cdot d$$ · spec"]
        NEA["<b>NASA Exoplanet Archive</b><br/>$$T_{\mathrm{eff}}\cdot R_\star\cdot M_\star$$<br/>insol · log g"]
        SIM["<b>SIMBAD</b><br/>$$T_{\mathrm{eff}}\cdot R_\star$$<br/>log g"]
    end

    %% ── ② EMC Catalog
    subgraph CAT["② EMC Catalog Columns"]
        direction LR
        R_J["$$r_p$$ (transit)"]
        MSI["$$m\sin i$$ (RV)"]
        A_C["$$a$$"]
        P_C["$$P$$"]
    end

    HPIC -->|"prio 1"| TEFF["$$T_{\mathrm{eff}}$$"]
    NEA -->|"prio 2"| TEFF
    SIM -->|"prio 3"| TEFF

    HPIC -->|"prio 1"| RSTAR["$$R_\star$$"]
    NEA -->|"prio 2"| RSTAR
    SIM -->|"prio 3"| RSTAR

    HPIC -->|"prio 1"| MSTAR["$$M_\star$$"]
    NEA -->|"prio 2"| MSTAR

    TEFF -. "if $$R_\star$$ missing" .-> ZAMS["<b>ZAMS fallback</b><br/>$$R_\star\approx(T_{\mathrm{eff}}/T_\odot)^{1.8}$$"]
    ZAMS -. "fallback" .-> RSTAR

    RSTAR -. " " .-> LUM["$$L_\star=R_\star^2(T_{\mathrm{eff}}/T_\odot)^4$$"]
    TEFF -.-> LUM

    MSTAR -. "if $$a$$ missing" .-> KEPL["<b>Kepler 3rd law fallback</b><br/>$$a=\bigl(M_\star(P/P_{\mathrm{yr}})^2\bigr)^{1/3}$$"]
    P_C -.-> KEPL
    A_C -->|"direct"| AFIN["$$a$$"]
    KEPL -. "fallback" .-> AFIN

    LUM -.-> FCALC["$$F=L_\star/a^2$$"]
    AFIN -.-> FCALC
    NEA -->|"direct insol"| FLUX["$$F_{\mathrm{rel}}$$"]
    FCALC -. "fallback" .-> FLUX

    R_J --> REARTH["$$r_p$$"]

    MSI -. "only when $$r_p$$ masked" .-> CK["<b>Chen &amp; Kipping [1]</b><br/>$$M\lt 2.04\,M_\oplus:\;R=1.008\,M^{0.279}$$<br/>$$2.04\text{--}132\,M_\oplus:\;R=0.808\,M^{0.589}$$<br/><br/>Isotropic incl. prior [2]:<br/>$$P(\sin i\gt\sin i_{\min})=\sqrt{1-\sin^2 i_{\min}}$$<br/>$$\sin i_{\min}=0.5\;\Rightarrow\;86.6\%\,\mathrm{CI}$$<br/>$$R_{\min}=f(m\sin i),\quad R_{\max}=f(m_{\max})$$"]
    CK -.-> RBOUNDS["$$R_{\min},\,R_{\max}$$<br/>(masked when $$r_p$$ known)"]

    subgraph CLASS["③ Classification Masks"]
        direction LR
        ROCKY[/"<b>rocky_mask()</b><br/>confirmed: $$r_p\in[0.5,1.5]\,R_\oplus$$<br/>uncertain: $$[R_{\min},R_{\max}]\cap[0.5,1.5]\neq\emptyset$$"/]
        TEMP[/"<b>temperate_mask()</b> [3]<br/>$$F_{\mathrm{rel}}\in[0.35,1.77]\,S_\oplus$$"/]
    end

    REARTH --> ROCKY
    RBOUNDS -. "interval check" .-> ROCKY
    FLUX --> TEMP
```

---
**References**  
[1] Chen & Kipping 2017, *ApJ* **834**, 17 — piecewise power-law mass–radius relation  
[2] Stevens & Gaudi 2013, *PASP* **125**, 933 — isotropic orbital inclination prior for RV detections  
[3] Kopparapu et al. 2013, *ApJ* **765**, 131 — conservative habitable-zone flux boundaries
