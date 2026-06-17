from __future__ import annotations
import re
import numpy as np

def classify_spectral_type(sptype: str) -> str:
    """Map a spectral type string to a broad stellar category.

    Returns one of: 'Sun-like', 'Low-luminosity', 'Very-low-luminosity', 'Other'.
    """
    s = str(sptype).strip()
    if s == 'null':
        return 'Other'
    if re.match(r'^D[A-Z0-9]', s):
        return 'Other'
    if re.search(r'IV|III|II', s):
        return 'Other'
    m = re.match(r'^d?([OBAFGKM])(\d+(?:\.\d+)?)', s, re.I)
    if not m:
        return 'Other'
    letter = m.group(1).upper()
    subtype = float(m.group(2))
    if letter in ('O', 'B', 'A'):
        return 'Other'
    if letter in ('F', 'G'):
        return 'Sun-like'
    if letter == 'K':
        return 'Sun-like' if subtype <= 5 else 'Low-luminosity'
    if letter == 'M':
        return 'Low-luminosity' if subtype < 3 else 'Very-low-luminosity'
    return 'Other'


def spectype_display(spec: str, teff: float) -> str:
    """Spectral type for display: use the actual string if available, else derive from teff."""
    s = str(spec).strip()
    if s and s != 'null':
        return s
    return teff_to_spectype(teff)

 
_LETTER_ORDER = "OBAFGKMLTY"
 
# (spectral_type, Teff_K) — strictly hot → cool
_TEFF_SPECTYPE: list[tuple[str, int]] = [
    # O  —  Martins+2005
    ("O3",  44200), ("O4",  42900), ("O5",  40900), ("O5.5", 39700),
    ("O6",  39500), ("O6.5",38100), ("O7",  37100), ("O7.5", 36100),
    ("O8",  35100), ("O8.5",34100), ("O9",  33100), ("O9.5", 32000),
    # B–M  —  Pecaut & Mamajek 2013
    ("B0",  29700), ("B1",  25400), ("B2",  22000), ("B3",  17600),
    ("B5",  15400), ("B6",  14500), ("B7",  13400), ("B8",  11400), ("B9", 10500),
    ("A0",   9600), ("A1",   9330), ("A2",   9040), ("A3",   8750),
    ("A4",   8480), ("A5",   8180), ("A7",   7920),
    ("F0",   7220), ("F2",   7030), ("F3",   6910), ("F5",   6640),
    ("F6",   6510), ("F7",   6340), ("F8",   6160),
    ("G0",   5930), ("G1",   5860), ("G2",   5780), ("G5",   5660), ("G8", 5440),
    ("K0",   5280), ("K1",   5170), ("K2",   5040), ("K3",   4830),
    ("K4",   4600), ("K5",   4410), ("K6",   4230), ("K7",   4070),
    ("K8",   3990), ("K9",   3930),
    ("M0",   3850), ("M1",   3660), ("M2",   3560), ("M3",   3410),
    ("M4",   3200), ("M5",   3030), ("M6",   2850), ("M7",   2650),
    ("M8",   2500), ("M9",   2400),
]
 
def _parse_spt(spt: str) -> float:
    """Spectral type string → float index (class_rank * 10 + subtype).
    Luminosity-class suffixes and peculiarity flags are ignored."""
    m = re.match(r"([OBAFGKMLTY])(\d+\.?\d*)?", spt.strip().upper())
    if m is None:
        raise ValueError(f"Cannot parse spectral type: {spt!r}")
    letter, num = m.group(1), m.group(2)
    return _LETTER_ORDER.index(letter) * 10 + (float(num) if num else 5.0)
 
 
_nums  = np.array([_parse_spt(s) for s, _ in _TEFF_SPECTYPE])  # strictly increasing
_teffs = np.array([t for _, t in _TEFF_SPECTYPE], dtype=float)  # strictly decreasing
 
 
def spectype_to_teff(spectype: str) -> float:
    """Interpolated Teff (K) for a main-sequence spectral type.
    Accepts fractional subtypes ('K2.5'), lum-class suffixes ('G2V'), and
    peculiarity flags. Raises ValueError on unparseable input.
    
    Sources
    O3–O9.5 : Martins, Schaerer & Hillier 2005, A&A 436, 1049 (Table 1, dwarf scale)
    B0–M9   : Pecaut & Mamajek 2013, ApJS 208, 9
            https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    
    """
    return float(np.interp(_parse_spt(spectype), _nums, _teffs))
 
 
def teff_to_spectype(teff: float, *, prefix: str = "~") -> str:
    """Nearest-anchor main-sequence spectral type for a given Teff.
    Returns '' for non-positive Teff. Output is always a discrete anchor
    entry; use spectype_to_teff() for the interpolated inverse.
    
    Sources
    O3–O9.5 : Martins, Schaerer & Hillier 2005, A&A 436, 1049 (Table 1, dwarf scale)
    B0–M9   : Pecaut & Mamajek 2013, ApJS 208, 9
            https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    
    
    """
    if not (teff > 0):
        return ""
    nearest = int(np.argmin(np.abs(_teffs - teff)))
    return f"{prefix}{_TEFF_SPECTYPE[nearest][0]}"


def get_spectral_class_range(spectype: str) -> tuple[float, float]:
    """Get the min and max Teff (K) for the major spectral class (letter) of spectype.

    Parses the provided spectral type string to identify its primary spectral class
    (e.g., 'O', 'B', 'A', 'F', 'G', 'K', 'M'), then returns the minimum and maximum 
    effective temperatures corresponding to that class based on the internal anchor tables.

    Parameters
    ----------
    spectype : str
        The full spectral type string (e.g., 'G2V', 'M5').

    Returns
    -------
    tuple[float, float]
        A tuple of (min_teff, max_teff). Returns (0.0, 0.0) if the spectral class 
        is invalid or not found.
    """
    m = re.search(r'([OBAFGKMLTY])', spectype.strip().upper())
    if not m:
        return 0.0, 0.0
        
    letter = m.group(1)
    class_teffs = [t for name, t in _TEFF_SPECTYPE if name.upper().startswith(letter)]
    
    if not class_teffs:
        return 0.0, 0.0
        
    return min(class_teffs), max(class_teffs)


